"""Bilevel planning models for the cluttered retrieval 2D environment."""

from typing import Sequence, cast

import numpy as np
from bilevel_planning.structs import (
    LiftedParameterizedController,
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from gymnasium.spaces import Space
from numpy.typing import NDArray
from prbench.envs.geom2d.clutteredretrieval2d import (
    ClutteredRetrieval2DEnvConfig,
    ObjectCentricClutteredRetrieval2DEnv,
    TargetBlockType,
)
from prbench.envs.geom2d.object_types import CRVRobotType, RectangleType
from prbench.envs.geom2d.structs import SE2Pose
from prbench.envs.geom2d.utils import (
    CRVRobotActionSpace,
    get_suctioned_objects,
    run_motion_planning_for_crv_robot,
    snap_suctioned_objects,
    state_2d_has_collision,
)
from prbench_models.geom2d.utils import (
    Geom2dRobotController,
)
from relational_structs import (
    GroundAtom,
    LiftedAtom,
    LiftedOperator,
    Object,
    ObjectCentricState,
    Predicate,
    Variable,
)
from relational_structs.spaces import ObjectCentricBoxSpace, ObjectCentricStateSpace


def create_bilevel_planning_models(
    observation_space: Space, action_space: Space, num_obstructions: int
) -> SesameModels:
    """Create the env models for cluttered retrieval 2D."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, CRVRobotActionSpace)

    sim = ObjectCentricClutteredRetrieval2DEnv(num_obstructions=num_obstructions)
    env_config = ClutteredRetrieval2DEnvConfig()
    world_x_min = env_config.world_min_x + env_config.robot_base_radius
    world_x_max = env_config.world_max_x - env_config.robot_base_radius
    world_y_min = env_config.world_min_y + env_config.robot_base_radius
    world_y_max = env_config.world_max_y - env_config.robot_base_radius

    # Convert observations into states. The important thing is that states are hashable.
    def observation_to_state(o: NDArray[np.float32]) -> ObjectCentricState:
        """Convert the vectors back into (hashable) object-centric states."""
        return observation_space.devectorize(o)

    # Create the transition function.
    def transition_fn(
        x: ObjectCentricState,
        u: NDArray[np.float32],
    ) -> ObjectCentricState:
        """Simulate the action."""
        state = x.copy()
        sim.reset(options={"init_state": state})
        obs, _, _, _, _ = sim.step(u)
        return obs.copy()

    # Types.
    types = {CRVRobotType, RectangleType, TargetBlockType}

    # Create the state space.
    state_space = ObjectCentricStateSpace(types)

    # Predicates.
    HoldingTgt = Predicate("HoldingTgt", [CRVRobotType, TargetBlockType])
    HoldingObstruction = Predicate("HoldingObstruction", [CRVRobotType, RectangleType])
    HandEmpty = Predicate("HandEmpty", [CRVRobotType])
    predicates = {HoldingTgt, HoldingObstruction, HandEmpty}

    # State abstractor.
    def state_abstractor(x: ObjectCentricState) -> RelationalAbstractState:
        """Get the abstract state for the current state."""
        robot = x.get_objects(CRVRobotType)[0]
        target_block = x.get_objects(TargetBlockType)[0]
        obstructions = x.get_objects(RectangleType)
        atoms: set[GroundAtom] = set()
        # Add holding / handempty atoms.
        suctioned_objs = {o for o, _ in get_suctioned_objects(x, robot)}
        # Check what the robot is holding
        if target_block in suctioned_objs:
            atoms.add(GroundAtom(HoldingTgt, [robot, target_block]))
        else:
            # Check if holding any obstruction
            held_obstruction = None
            for obstruction in obstructions:
                if obstruction in suctioned_objs:
                    held_obstruction = obstruction
                    break

            if held_obstruction is not None:
                atoms.add(GroundAtom(HoldingObstruction, [robot, held_obstruction]))
            else:
                atoms.add(GroundAtom(HandEmpty, [robot]))
        objects = {robot, target_block} | set(obstructions)
        return RelationalAbstractState(atoms, objects)

    # Goal abstractor.
    def goal_deriver(x: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to retrieve the target block (hold it)."""
        del x  # not needed
        robot = Object("robot", CRVRobotType)
        target_block = Object("target_block", TargetBlockType)
        atoms = {GroundAtom(HoldingTgt, [robot, target_block])}
        return RelationalAbstractGoal(atoms, state_abstractor)

    # Operators.
    robot = Variable("?robot", CRVRobotType)
    target_block = Variable("?target_block", TargetBlockType)
    obstruction = Variable("?obstruction", RectangleType)

    PickTgtOperator = LiftedOperator(
        "PickTgt",
        [robot, target_block],
        preconditions={LiftedAtom(HandEmpty, [robot])},
        add_effects={LiftedAtom(HoldingTgt, [robot, target_block])},
        delete_effects={LiftedAtom(HandEmpty, [robot])},
    )

    PickObstructionOperator = LiftedOperator(
        "PickObstruction",
        [robot, obstruction],
        preconditions={LiftedAtom(HandEmpty, [robot])},
        add_effects={LiftedAtom(HoldingObstruction, [robot, obstruction])},
        delete_effects={LiftedAtom(HandEmpty, [robot])},
    )

    PlaceObstructionOperator = LiftedOperator(
        "PlaceObstruction",
        [robot, obstruction],
        preconditions={LiftedAtom(HoldingObstruction, [robot, obstruction])},
        add_effects={LiftedAtom(HandEmpty, [robot])},
        delete_effects={LiftedAtom(HoldingObstruction, [robot, obstruction])},
    )

    # Controllers.
    class GroundPickController(Geom2dRobotController):
        """Controller for picking rectangular objects (target blocks or obstructions).

        The grasping point is sampled on all four sides of the block.
        """

        def __init__(self, objects: Sequence[Object]) -> None:
            assert isinstance(action_space, CRVRobotActionSpace)
            super().__init__(objects, action_space)
            self._block = objects[1]
            assert self._block.is_instance(TargetBlockType) or self._block.is_instance(
                RectangleType
            )

        def sample_parameters(
            self, x: ObjectCentricState, rng: np.random.Generator
        ) -> tuple[float, float, float]:
            # Sample grasp ratio and side
            # grasp_ratio: determines position along the side ([0.0, 1.0])
            # side: 0~0.25 left, 0.25~0.5 right, 0.5~0.75 top, 0.75~1.0 bottom
            full_state = x.copy()
            init_constant_state = sim.initial_constant_state
            if init_constant_state is not None:
                full_state.data.update(init_constant_state.data)
            while True:
                grasp_ratio = rng.uniform(0.0, 1.0)
                side = rng.uniform(0.0, 1.0)
                max_arm_length = x.get(self._robot, "arm_length")
                min_arm_length = (
                    x.get(self._robot, "base_radius")
                    + x.get(self._robot, "gripper_width") / 2
                    + 1e-4
                )
                arm_length = rng.uniform(min_arm_length, max_arm_length)
                # Calcuate Robot Pos
                target_se2_pose = self._calculate_grasp_robot_pose(
                    x, grasp_ratio, side, arm_length
                )
                # Check if the target pose is collision-free
                full_state.set(self._robot, "x", target_se2_pose.x)
                full_state.set(self._robot, "y", target_se2_pose.y)
                full_state.set(self._robot, "theta", target_se2_pose.theta)
                full_state.set(self._robot, "arm_joint", arm_length)
                # Check collision
                moving_objects = {self._robot}
                static_objects = set(full_state) - moving_objects
                if not state_2d_has_collision(
                    full_state, moving_objects, static_objects, {}
                ):
                    break

            # Pack parameters: side determines grasp approach, ratio determines position
            return (grasp_ratio, side, arm_length)

        def _get_vacuum_actions(self) -> tuple[float, float]:
            return 0.0, 1.0

        def _calculate_grasp_robot_pose(
            self,
            state: ObjectCentricState,
            ratio: float,
            side: float,
            arm_length: float,
        ) -> SE2Pose:
            """Calculate the grasp point based on side and ratio parameters."""
            # Get block properties
            block_x = state.get(self._block, "x")
            block_y = state.get(self._block, "y")
            block_theta = state.get(self._block, "theta")
            block_width = state.get(self._block, "width")
            block_height = state.get(self._block, "height")

            # Calculate reference point and approach direction based on side
            if side < 0.25:  # left side
                custom_dx = -(arm_length + state.get(self._robot, "gripper_width"))
                custom_dy = ratio * block_height
                custom_dtheta = 0.0
            elif 0.25 <= side < 0.5:  # right side
                custom_dx = (
                    arm_length + state.get(self._robot, "gripper_width") + block_width
                )
                custom_dy = ratio * block_height
                custom_dtheta = np.pi
            elif 0.5 <= side < 0.75:  # top side
                custom_dx = ratio * block_width
                custom_dy = (
                    arm_length + state.get(self._robot, "gripper_width") + block_height
                )
                custom_dtheta = -np.pi / 2
            else:  # bottom side
                custom_dx = ratio * block_width
                custom_dy = -(arm_length + state.get(self._robot, "gripper_width"))
                custom_dtheta = np.pi / 2

            target_se2_pose = SE2Pose(block_x, block_y, block_theta) * SE2Pose(
                custom_dx, custom_dy, custom_dtheta
            )
            return target_se2_pose

        def _generate_waypoints(
            self, state: ObjectCentricState
        ) -> list[tuple[SE2Pose, float]]:
            """Generate waypoints to the grasp point."""
            params = cast(tuple[float, ...], self._current_params)
            grasp_ratio = params[0]
            side = params[1]
            desired_arm_length = params[2]
            robot_x = state.get(self._robot, "x")
            robot_y = state.get(self._robot, "y")
            robot_theta = state.get(self._robot, "theta")
            robot_radius = state.get(self._robot, "base_radius")
            # Calculate grasp point and robot target position
            target_se2_pose = self._calculate_grasp_robot_pose(
                state, grasp_ratio, side, desired_arm_length
            )

            # Plan collision-free waypoints to the target pose
            mp_state = state.copy()
            mp_state.set(self._robot, "arm_joint", robot_radius)
            init_constant_state = sim.initial_constant_state
            if init_constant_state is not None:
                mp_state.data.update(init_constant_state.data)
            assert isinstance(action_space, CRVRobotActionSpace)
            collision_free_waypoints = run_motion_planning_for_crv_robot(
                mp_state, self._robot, target_se2_pose, action_space
            )
            # Always first make arm shortest to avoid collisions
            final_waypoints: list[tuple[SE2Pose, float]] = [
                (SE2Pose(robot_x, robot_y, robot_theta), robot_radius)
            ]

            if collision_free_waypoints is not None:
                for wp in collision_free_waypoints:
                    final_waypoints.append((wp, robot_radius))
                final_waypoints.append((target_se2_pose, desired_arm_length))
                return final_waypoints
            # If motion planning fails, raise failure
            raise TrajectorySamplingFailure(
                "Failed to find a collision-free path to target."
            )

    class GroundPlaceController(Geom2dRobotController):
        """Controller for placing rectangular objects (target blocks or obstructions) in
        a collision-free location."""

        def __init__(self, objects: Sequence[Object]) -> None:
            assert isinstance(action_space, CRVRobotActionSpace)
            super().__init__(objects, action_space)
            self._block = objects[1]
            assert self._block.is_instance(TargetBlockType) or self._block.is_instance(
                RectangleType
            )

        def sample_parameters(
            self, x: ObjectCentricState, rng: np.random.Generator
        ) -> tuple[float, float, float]:
            # Sample collision-free robot pose
            full_state = x.copy()
            init_constant_state = sim.initial_constant_state
            if init_constant_state is not None:
                full_state.data.update(init_constant_state.data)
            while True:
                abs_x = rng.uniform(world_x_min, world_x_max)
                abs_y = rng.uniform(world_y_min, world_y_max)
                abs_theta = rng.uniform(-np.pi, np.pi)
                full_state.set(self._robot, "x", abs_x)
                full_state.set(self._robot, "y", abs_y)
                full_state.set(self._robot, "theta", abs_theta)
                suctioned_objects = get_suctioned_objects(x, self._robot)
                snap_suctioned_objects(full_state, self._robot, suctioned_objects)
                # Check collision
                moving_objects = {self._robot} | {o for o, _ in suctioned_objects}
                static_objects = set(full_state) - moving_objects
                if not state_2d_has_collision(
                    full_state, moving_objects, static_objects, {}
                ):
                    break
            rel_x = (abs_x - world_x_min) / (world_x_max - world_x_min)
            rel_y = (abs_y - world_y_min) / (world_y_max - world_y_min)
            rel_theta = (abs_theta + np.pi) / (2 * np.pi)

            return (rel_x, rel_y, rel_theta)

        def _get_vacuum_actions(self) -> tuple[float, float]:
            return 1.0, 0.0

        def _generate_waypoints(
            self, state: ObjectCentricState
        ) -> list[tuple[SE2Pose, float]]:
            robot_x = state.get(self._robot, "x")
            robot_y = state.get(self._robot, "y")
            robot_theta = state.get(self._robot, "theta")
            robot_radius = state.get(self._robot, "base_radius")
            # Calculate place position
            params = cast(tuple[float, ...], self._current_params)
            final_robot_x = world_x_min + (world_x_max - world_x_min) * params[0]
            final_robot_y = world_y_min + (world_y_max - world_y_min) * params[1]
            final_robot_theta = -np.pi + (2 * np.pi) * params[2]
            final_robot_pose = SE2Pose(final_robot_x, final_robot_y, final_robot_theta)

            current_wp = (
                SE2Pose(robot_x, robot_y, robot_theta),
                robot_radius,
            )
            # Plan collision-free waypoints to the target pose
            # We set the arm to be the longest during motion planning
            final_waypoints: list[tuple[SE2Pose, float]] = [current_wp]
            mp_state = state.copy()
            mp_state.set(self._robot, "arm_joint", robot_radius)
            init_constant_state = sim.initial_constant_state
            if init_constant_state is not None:
                mp_state.data.update(init_constant_state.data)
            assert isinstance(action_space, CRVRobotActionSpace)
            collision_free_waypoints_0 = run_motion_planning_for_crv_robot(
                mp_state, self._robot, final_robot_pose, action_space
            )
            if collision_free_waypoints_0 is None:
                # Stay static
                raise TrajectorySamplingFailure(
                    "Failed to find a collision-free path to target."
                )
            for wp in collision_free_waypoints_0:
                final_waypoints.append((wp, robot_radius))

            return final_waypoints

    # Lifted controllers.
    PickTgtController: LiftedParameterizedController = LiftedParameterizedController(
        [robot, target_block],
        GroundPickController,
    )

    PickObstructionController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, obstruction],
            GroundPickController,
        )
    )

    PlaceObstructionController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, obstruction],
            GroundPlaceController,
        )
    )

    # Finalize the skills.
    skills = {
        LiftedSkill(PickTgtOperator, PickTgtController),
        LiftedSkill(PickObstructionOperator, PickObstructionController),
        LiftedSkill(PlaceObstructionOperator, PlaceObstructionController),
    }

    # Finalize the models.
    return SesameModels(
        observation_space,
        state_space,
        action_space,
        transition_fn,
        types,
        predicates,
        observation_to_state,
        state_abstractor,
        goal_deriver,
        skills,
    )
