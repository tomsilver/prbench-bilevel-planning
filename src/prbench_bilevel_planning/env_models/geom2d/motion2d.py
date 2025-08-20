"""Bilevel planning models for the motion 2D environment."""

from typing import Any, Sequence, cast

import numpy as np
from bilevel_planning.structs import (
    LiftedParameterizedController,
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from geom2drobotenvs.object_types import CRVRobotType
from geom2drobotenvs.structs import SE2Pose
from geom2drobotenvs.utils import (
    CRVRobotActionSpace,
    rectangle_object_to_geom,
    run_motion_planning_for_crv_robot,
)
from gymnasium.spaces import Space
from numpy.typing import NDArray
from prbench.envs.geom2d.motion2d import (
    RectangleType,
    ObjectCentricMotion2DEnv,
    TargetRegionType,
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

from prbench_bilevel_planning.env_models.geom2d.geom2d_utils import (
    Geom2dRobotController,
)


def create_bilevel_planning_models(
    observation_space: Space, action_space: Space, num_passages: int = 2
) -> SesameModels:
    """Create the env models for motion 2D."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, CRVRobotActionSpace)

    sim = ObjectCentricMotion2DEnv(num_passages=num_passages)

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
    types = {CRVRobotType, TargetRegionType, RectangleType}

    # Create the state space.
    state_space = ObjectCentricStateSpace(types)

    # Predicates.
    At = Predicate("At", [CRVRobotType, TargetRegionType])
    NotAt = Predicate("NotAt", [CRVRobotType, TargetRegionType])
    predicates = {At, NotAt}

    # State abstractor.
    def state_abstractor(x: ObjectCentricState) -> RelationalAbstractState:
        """Get the abstract state for the current state."""
        robot = x.get_objects(CRVRobotType)[0]
        target_region = x.get_objects(TargetRegionType)[0]

        atoms: set[GroundAtom] = set()
        
        # Check if robot is in the target region
        robot_x = x.get(robot, "x")
        robot_y = x.get(robot, "y")
        target_region_geom = rectangle_object_to_geom(x, target_region, {})
        
        if target_region_geom.contains_point(robot_x, robot_y):
            atoms.add(GroundAtom(At, [robot, target_region]))
        else:
            atoms.add(GroundAtom(NotAt, [robot, target_region]))
            
        objects = {robot, target_region}
        return RelationalAbstractState(atoms, objects)

    # Goal abstractor.
    def goal_deriver(x: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to have the robot at the target region."""
        robot = x.get_objects(CRVRobotType)[0]
        target_region = x.get_objects(TargetRegionType)[0]
        atoms = {GroundAtom(At, [robot, target_region])}
        return RelationalAbstractGoal(atoms, state_abstractor)

    # Operators.
    robot = Variable("?robot", CRVRobotType)
    target = Variable("?target", TargetRegionType)

    MoveToOperator = LiftedOperator(
        "MoveTo",
        [robot, target],
        preconditions={
            LiftedAtom(NotAt, [robot, target]),
        },
        add_effects={LiftedAtom(At, [robot, target])},
        delete_effects={LiftedAtom(NotAt, [robot, target])},
    )

    # Controllers.
    class GroundMoveToController(Geom2dRobotController):
        """Controller for moving the robot to the target region."""

        def __init__(self, objects: Sequence[Object]) -> None:
            assert isinstance(action_space, CRVRobotActionSpace)
            super().__init__(objects, action_space)
            self._target = objects[1]

        def sample_parameters(
            self, x: ObjectCentricState, rng: np.random.Generator
        ) -> tuple[float, float, float]:
            # Sample a point within the target region and a random orientation
            target_x = x.get(self._target, "x")
            target_y = x.get(self._target, "y")
            target_width = x.get(self._target, "width")
            target_height = x.get(self._target, "height")
            
            # Sample relative position within the target region
            rel_x = rng.uniform(0.1, 0.9)
            rel_y = rng.uniform(0.1, 0.9)
            
            # Convert to absolute coordinates within target bounds
            abs_x = target_x + rel_x * target_width
            abs_y = target_y + rel_y * target_height
            
            # Sample random orientation
            abs_theta = rng.uniform(-np.pi, np.pi)
            
            return (rel_x, rel_y, abs_theta)

        def _get_vacuum_actions(self) -> tuple[float, float]:
            return 0.0, 0.0  # No vacuum needed for motion

        def _generate_waypoints(
            self, state: ObjectCentricState
        ) -> list[tuple[SE2Pose, float]]:
            robot_x = state.get(self._robot, "x")
            robot_y = state.get(self._robot, "y")
            robot_theta = state.get(self._robot, "theta")
            robot_radius = state.get(self._robot, "base_radius")
            
            # Calculate target position from parameters
            params = cast(tuple[float, float, float], self._current_params)
            target_x = state.get(self._target, "x")
            target_y = state.get(self._target, "y")
            target_width = state.get(self._target, "width")
            target_height = state.get(self._target, "height")
            
            final_x = target_x + params[0] * target_width
            final_y = target_y + params[1] * target_height
            final_theta = params[2]
            final_pose = SE2Pose(final_x, final_y, final_theta)

            current_wp = (
                SE2Pose(robot_x, robot_y, robot_theta),
                robot_radius,
            )

            # Use motion planning to find collision-free path
            assert isinstance(action_space, CRVRobotActionSpace)
            collision_free_waypoints = run_motion_planning_for_crv_robot(
                state, self._robot, final_pose, action_space
            )
            
            final_waypoints: list[tuple[SE2Pose, float]] = []
            
            if collision_free_waypoints is not None:
                for wp in collision_free_waypoints:
                    final_waypoints.append((wp, robot_radius))
            else:
                # If motion planning fails, stay in place
                final_waypoints.append(current_wp)

            return final_waypoints

    # Lifted controllers.
    MoveToController: LiftedParameterizedController = LiftedParameterizedController(
        [robot, target],
        GroundMoveToController,
    )

    # Finalize the skills.
    skills = {
        LiftedSkill(MoveToOperator, MoveToController),
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