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
    state_has_collision,
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
    AtTgt = Predicate("AtTgt", [CRVRobotType, TargetRegionType])
    NotAtTgt = Predicate("NotAtTgt", [CRVRobotType, TargetRegionType])
    AtPassage = Predicate("AtPassage", [CRVRobotType, RectangleType, RectangleType])
    NotAtPassage = Predicate("NotAtPassage", [CRVRobotType, RectangleType, RectangleType])
    NotAtAnyPassage = Predicate("NotAtAnyPassage", [CRVRobotType])
    predicates = {AtTgt, NotAtTgt, AtPassage, NotAtPassage, NotAtAnyPassage}

    # State abstractor.
    def state_abstractor(x: ObjectCentricState) -> RelationalAbstractState:
        """Get the abstract state for the current state."""
        robot = x.get_objects(CRVRobotType)[0]
        target_region = x.get_objects(TargetRegionType)[0]
        obstacles = x.get_objects(RectangleType)

        atoms: set[GroundAtom] = set()
        
        # Check if robot is in the target region
        robot_x = x.get(robot, "x")
        robot_y = x.get(robot, "y")
        target_region_geom = rectangle_object_to_geom(x, target_region, {})
        
        if target_region_geom.contains_point(robot_x, robot_y):
            atoms.add(GroundAtom(AtTgt, [robot, target_region]))
        else:
            atoms.add(GroundAtom(NotAtTgt, [robot, target_region]))

        robot_x = x.get(robot, "x")
        robot_radius = x.get(robot, "base_radius")
        robot_y = x.get(robot, "y")
        at_any_passage = False
        for obs1 in obstacles:
            for obs2 in obstacles:
                if obs1 != obs2:
                    obs1_x = x.get(obs1, "x")
                    obs2_x = x.get(obs2, "x")
                    if obs1_x == obs2_x:
                        obs_width = x.get(obs1, "width")
                        obs1_y = x.get(obs1, "y")
                        obs2_y = x.get(obs2, "y")
                        obs2_height = x.get(obs2, "height")
                        if obs1_y < obs2_y:
                            # Obstacle1 should be higher than obstacle2
                            continue
                        y_min = obs2_y + obs2_height
                        y_max = obs1_y
                        assert y_min <= y_max, "Obstacles should not overlap vertically"
                        x_close = abs(obs1_x + obs_width / 2 - robot_x) \
                            < robot_radius
                        y_close = (y_min <= robot_y <= y_max)
                        if x_close and y_close:
                            at_any_passage = True
                            atoms.add(GroundAtom(AtPassage, [robot, obs1, obs2]))
                        else:
                            atoms.add(GroundAtom(NotAtPassage, [robot, obs1, obs2]))
        if not at_any_passage:
            atoms.add(GroundAtom(NotAtAnyPassage, [robot]))   
            
        objects = {robot, target_region} | set(obstacles)
        return RelationalAbstractState(atoms, objects)

    # Goal abstractor.
    def goal_deriver(x: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to have the robot at the target region."""
        robot = x.get_objects(CRVRobotType)[0]
        target_region = x.get_objects(TargetRegionType)[0]
        atoms = {GroundAtom(AtTgt, [robot, target_region])}
        return RelationalAbstractGoal(atoms, state_abstractor)

    # Operators.
    robot = Variable("?robot", CRVRobotType)
    target = Variable("?target", TargetRegionType)
    obstacle1 = Variable("?obstacle1", RectangleType)
    obstacle2 = Variable("?obstacle2", RectangleType)
    obstacle3 = Variable("?obstacle3", RectangleType)
    obstacle4 = Variable("?obstacle4", RectangleType)

    MoveToTgtFromNoPassageOperator = LiftedOperator(
        "MoveToTgtFromNoPassage",
        [robot, target],
        preconditions={
            LiftedAtom(NotAtTgt, [robot, target]),
            LiftedAtom(NotAtAnyPassage, [robot]),
        },
        add_effects={LiftedAtom(AtTgt, [robot, target])},
        delete_effects={LiftedAtom(NotAtTgt, [robot, target])},
    )

    MoveToTgtFromPassageOperator = LiftedOperator(
        "MoveToTgtFromPassage",
        [robot, target, obstacle1, obstacle2],
        preconditions={
            LiftedAtom(NotAtTgt, [robot, target]),
            LiftedAtom(AtPassage, [robot, obstacle1, obstacle2]),
        },
        add_effects={LiftedAtom(AtTgt, [robot, target]),
                    LiftedAtom(NotAtAnyPassage, [robot]),
                    LiftedAtom(NotAtPassage, [robot, obstacle1, obstacle2])},
        delete_effects={LiftedAtom(NotAtTgt, [robot, target]),
                        LiftedAtom(AtPassage, [robot, obstacle1, obstacle2])},
    )

    MoveToPassageFromNoPassageOperator = LiftedOperator(
        "MoveToPassageFromNoPassage",
        [robot, obstacle1, obstacle2],
        preconditions={
            LiftedAtom(NotAtAnyPassage, [robot]),
            LiftedAtom(NotAtPassage, [robot, obstacle1, obstacle2]),
        },
        add_effects={LiftedAtom(AtPassage, [robot, obstacle1, obstacle2])},
        delete_effects={LiftedAtom(NotAtPassage, [robot, obstacle1, obstacle2]),
                        LiftedAtom(NotAtAnyPassage, [robot])},
    )

    MoveToPassageFromPassageOperator = LiftedOperator(
        "MoveToPassageFromPassage",
        [robot, obstacle1, obstacle2, obstacle3, obstacle4],
        preconditions={
            LiftedAtom(AtPassage, [robot, obstacle1, obstacle2]),
            LiftedAtom(NotAtPassage, [robot, obstacle3, obstacle4]),
        },
        add_effects={LiftedAtom(AtPassage, [robot, obstacle3, obstacle4])},
        delete_effects={LiftedAtom(NotAtPassage, [robot, obstacle3, obstacle4]),
                        LiftedAtom(AtPassage, [robot, obstacle1, obstacle2])},
    )

    # Controllers.
    class GroundMoveToTgtController(Geom2dRobotController):
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
            full_state = x.copy()
            init_constant_state = sim.initial_constant_state
            if init_constant_state is not None:
                full_state.data.update(init_constant_state.data)
            while True:
                # Sample relative position within the target region
                rel_x = rng.uniform(0.1, 0.9)
                rel_y = rng.uniform(0.1, 0.9)
                # Sample random orientation
                abs_theta = rng.uniform(-np.pi, np.pi)

                # Convert to absolute coordinates within target bounds
                abs_x = target_x + rel_x * target_width
                abs_y = target_y + rel_y * target_height
                full_state.set(self._robot, "x", abs_x)
                full_state.set(self._robot, "y", abs_y)
                full_state.set(self._robot, "theta", abs_theta)
                # Check collision
                moving_objects = {self._robot}
                static_objects = set(full_state) - moving_objects
                if not state_has_collision(
                    full_state, moving_objects, static_objects, {}
                ):
                    break
            # Relative orientation
            rel_theta = (abs_theta + np.pi) / (2 * np.pi)
            
            return (rel_x, rel_y, rel_theta)

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
            # Convert to absolute angle
            final_theta = params[2] * 2 * np.pi - np.pi 
            final_pose = SE2Pose(final_x, final_y, final_theta)

            current_wp = (
                SE2Pose(robot_x, robot_y, robot_theta),
                robot_radius,
            )

            # Use motion planning to find collision-free path
            assert isinstance(action_space, CRVRobotActionSpace)
            collision_free_waypoints = run_motion_planning_for_crv_robot(
                state, self._robot, final_pose, action_space)
            
            final_waypoints: list[tuple[SE2Pose, float]] = []
            
            if collision_free_waypoints is not None:
                for wp in collision_free_waypoints:
                    final_waypoints.append((wp, robot_radius))
            else:
                # If motion planning fails, stay in place
                final_waypoints.append(current_wp)

            return final_waypoints

    class GroundMoveToPassageController(GroundMoveToTgtController):
        """Controller for moving the robot to a passage."""

        def __init__(self, objects: Sequence[Object]) -> None:
            assert isinstance(action_space, CRVRobotActionSpace)
            super().__init__(objects)
            self._obstacle1 = objects[1]
            self._obstacle2 = objects[2]

        def sample_parameters(
            self, x: ObjectCentricState, rng: np.random.Generator
        ) -> tuple[float, float]:
            # Sample a point between the two obstacles
            obstacle1_x = x.get(self._obstacle1, "x")
            obstacle2_x = x.get(self._obstacle2, "x")
            obstacle1_width = x.get(self._obstacle1, "width")
            obstacle1_y = x.get(self._obstacle1, "y")
            obstacle2_y = x.get(self._obstacle2, "y")
            obstacle2_height = x.get(self._obstacle2, "height")
            robot_radius = x.get(self._robot, "base_radius")
            full_state = x.copy()
            init_constant_state = sim.initial_constant_state
            if init_constant_state is not None:
                full_state.data.update(init_constant_state.data)
            while True:
                rel_x = rng.uniform(0.1, 0.9)
                rel_y = rng.uniform(0.1, 0.9)

                abs_x = obstacle1_x + obstacle1_width / 2 - robot_radius + \
                    2 * robot_radius * rel_x
                rel_x * (obstacle2_x - obstacle1_x)
                abs_y = obstacle1_y + rel_y * (obstacle2_y - obstacle1_y)

                full_state.set(self._robot, "x", abs_x)
                full_state.set(self._robot, "y", abs_y)

                # Check collision
                moving_objects = {self._robot}
                static_objects = set(full_state) - moving_objects
                if not state_has_collision(
                    full_state, moving_objects, static_objects, {}
                ):
                    break

            return (rel_x, rel_y)

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
            # Convert to absolute angle
            final_theta = params[2] * 2 * np.pi - np.pi 
            final_pose = SE2Pose(final_x, final_y, final_theta)

            current_wp = (
                SE2Pose(robot_x, robot_y, robot_theta),
                robot_radius,
            )

            # Use motion planning to find collision-free path
            assert isinstance(action_space, CRVRobotActionSpace)
            collision_free_waypoints = run_motion_planning_for_crv_robot(
                state, self._robot, final_pose, action_space)
            
            final_waypoints: list[tuple[SE2Pose, float]] = []
            
            if collision_free_waypoints is not None:
                for wp in collision_free_waypoints:
                    final_waypoints.append((wp, robot_radius))
            else:
                # If motion planning fails, stay in place
                final_waypoints.append(current_wp)

            return final_waypoints


    # Lifted controllers.
    MoveToTgtFromNoPassageController: LiftedParameterizedController = LiftedParameterizedController(
        [robot, target],
        GroundMoveToTgtController,
    )
    MoveToTgtFromPassageController: LiftedParameterizedController = LiftedParameterizedController(
        [robot, target, obstacle1, obstacle2],
        GroundMoveToTgtController,
    )
    MoveToPassageFromNoPassageController: LiftedParameterizedController = LiftedParameterizedController(
        [robot, obstacle1, obstacle2],
        GroundMoveToPassageController,
    )
    MoveToPassageFromPassageController: LiftedParameterizedController = LiftedParameterizedController(
        [robot, obstacle1, obstacle2, obstacle3, obstacle4],
        GroundMoveToPassageController,
    )

    # Finalize the skills.
    skills = {
        LiftedSkill(MoveToTgtFromNoPassageOperator, MoveToTgtFromNoPassageController),
        LiftedSkill(MoveToTgtFromPassageOperator, MoveToTgtFromPassageController),
        LiftedSkill(MoveToPassageFromNoPassageOperator, MoveToPassageFromNoPassageController),
        LiftedSkill(MoveToPassageFromPassageOperator, MoveToPassageFromPassageController),
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