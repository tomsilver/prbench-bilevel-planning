"""Bilevel planning models for the stick button 2D environment."""

import abc
from typing import Sequence

import numpy as np
from bilevel_planning.structs import (
    GroundParameterizedController,
    LiftedParameterizedController,
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from geom2drobotenvs.object_types import CircleType, CRVRobotType, RectangleType
from geom2drobotenvs.utils import (
    CRVRobotActionSpace,
    get_suctioned_objects,
    object_to_multibody2d,
)
from gymnasium.spaces import Space
from numpy.typing import NDArray
from prbench.envs.stickbutton2d import (
    ObjectCentricStickButton2DEnv,
    StickButton2DEnvSpec,
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
from tomsgeoms2d.structs import Geom2D


def create_bilevel_planning_models(
    observation_space: Space, action_space: Space, num_buttons: int
) -> SesameModels:
    """Create the env models for stick button 2D."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, CRVRobotActionSpace)

    # Make a local copy of the environment to use as the "simulator". Note that we use
    # the object-centric version of the environment because we want access to the reset
    # and step functions in there, which operate over ObjectCentricState, which we use
    # as the state representation for planning.
    sim = ObjectCentricStickButton2DEnv(num_buttons=num_buttons)

    # Convert observations into states. The important thing is that states are hashable.
    def observation_to_state(o: NDArray[np.float32]) -> ObjectCentricState:
        """Convert the vectors back into (hashable) object-centric states."""
        return observation_space.devectorize(o)

    # The object-centric states that are passed around in planning do not include the
    # globally constant objects, so we need to create an exemplar state that does
    # include them and then copy in the changing values before calling step().
    exemplar_state = sim.reset()[0]

    # Create the transition function.
    def transition_fn(
        x: ObjectCentricState, u: NDArray[np.float32]
    ) -> ObjectCentricState:
        """Simulate the action."""
        # See note above re: why we can't just sim.reset(options={"init_state": x}).
        state = exemplar_state.copy()
        for obj, feats in x.data.items():
            state.data[obj] = feats
        # Now we can reset().
        sim.reset(options={"init_state": state})
        sim_obs, _, _, _, _ = sim.step(u)

        # Now we need to extract back out the changing objects.
        next_x = x.copy()
        for obj in x:
            next_x.data[obj] = sim_obs.data[obj]
        return next_x

    # Types.
    types = {CRVRobotType, RectangleType, CircleType}

    # Create the state space.
    state_space = ObjectCentricStateSpace(types)

    # Predicates.
    Grasped = Predicate("Grasped", [CRVRobotType, RectangleType])
    HandEmpty = Predicate("HandEmpty", [CRVRobotType])
    Pressed = Predicate("Pressed", [CircleType])
    RobotAboveButton = Predicate("RobotAboveButton", [CRVRobotType, CircleType])
    StickAboveButton = Predicate("StickAboveButton", [RectangleType, CircleType])
    AboveNoButton = Predicate("AboveNoButton", [])
    predicates = {
        Grasped,
        HandEmpty,
        Pressed,
        RobotAboveButton,
        StickAboveButton,
        AboveNoButton,
    }

    # State abstractor.
    def state_abstractor(x: ObjectCentricState) -> RelationalAbstractState:
        """Get the abstract state for the current state."""
        robot = x.get_objects(CRVRobotType)[0]
        stick = x.get_objects(RectangleType)[0]
        buttons = x.get_objects(CircleType)

        atoms: set[GroundAtom] = set()

        # Add grasped / handempty atoms.
        suctioned_objs = {o for o, _ in get_suctioned_objects(x, robot)}
        if stick in suctioned_objs:
            atoms.add(GroundAtom(Grasped, [robot, stick]))
        else:
            atoms.add(GroundAtom(HandEmpty, [robot]))

        # Get button colors from spec
        spec = StickButton2DEnvSpec()

        # Add button status atoms based on color.
        for button in buttons:
            button_color = (
                x.get(button, "color_r"),
                x.get(button, "color_g"),
                x.get(button, "color_b"),
            )
            if np.allclose(button_color, spec.button_pressed_rgb, atol=1e-3):
                atoms.add(GroundAtom(Pressed, [button]))

        # Add spatial relationship atoms using geometry
        robot_multi_body = object_to_multibody2d(robot, x, {})
        stick_multi_body = object_to_multibody2d(stick, x, {})
        assert len(stick_multi_body.bodies) == 1
        stick_geom = stick_multi_body.bodies[0].geom
        robot_geom: set[Geom2D] = set()
        for body in robot_multi_body.bodies:
            robot_geom.add(body.geom)

        robot_above_any_button = False
        stick_above_any_button = False

        for button in buttons:
            button_multi_body = object_to_multibody2d(button, x, {})
            assert len(button_multi_body.bodies) == 1
            button_geom = button_multi_body.bodies[0].geom

            # Check if robot is above button (geometrically intersects)
            if any(
                robot_geom_sub.intersects(button_geom) for robot_geom_sub in robot_geom
            ):
                atoms.add(GroundAtom(RobotAboveButton, [robot, button]))
                robot_above_any_button = True

            # Check if stick is above button (geometrically intersects)
            if stick_geom.intersects(button_geom):
                atoms.add(GroundAtom(StickAboveButton, [stick, button]))
                stick_above_any_button = True

        # Add AboveNoButton if neither robot nor stick is above any button
        if not robot_above_any_button and not stick_above_any_button:
            atoms.add(GroundAtom(AboveNoButton, []))

        objects = {robot, stick} | set(buttons)
        return RelationalAbstractState(atoms, objects)

    # Goal abstractor.
    def goal_deriver(x: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to press all buttons."""
        del x  # not needed
        atoms: set[GroundAtom] = set()
        for i in range(num_buttons):
            button = Object(f"button{i}", CircleType)
            atoms.add(GroundAtom(Pressed, [button]))
        return RelationalAbstractGoal(atoms, state_abstractor)

    # Operators.
    robot = Variable("?robot", CRVRobotType)
    stick = Variable("?stick", RectangleType)
    button = Variable("?button", CircleType)
    from_button = Variable("?from_button", CircleType)

    # RobotPressButtonFromNothing
    RobotPressButtonFromNothingOperator = LiftedOperator(
        "RobotPressButtonFromNothing",
        [robot, button],
        preconditions={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(AboveNoButton, []),
        },
        add_effects={
            LiftedAtom(Pressed, [button]),
            LiftedAtom(RobotAboveButton, [robot, button]),
        },
        delete_effects={LiftedAtom(AboveNoButton, [])},
    )

    # RobotPressButtonFromButton
    RobotPressButtonFromButtonOperator = LiftedOperator(
        "RobotPressButtonFromButton",
        [robot, button, from_button],
        preconditions={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(RobotAboveButton, [robot, from_button]),
        },
        add_effects={
            LiftedAtom(Pressed, [button]),
            LiftedAtom(RobotAboveButton, [robot, button]),
        },
        delete_effects={LiftedAtom(RobotAboveButton, [robot, from_button])},
    )

    # PickStickFromNothing
    PickStickFromNothingOperator = LiftedOperator(
        "PickStickFromNothing",
        [robot, stick],
        preconditions={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(AboveNoButton, []),
        },
        add_effects={
            LiftedAtom(Grasped, [robot, stick]),
        },
        delete_effects={LiftedAtom(HandEmpty, [robot])},
    )

    # PickStickFromButton
    PickStickFromButtonOperator = LiftedOperator(
        "PickStickFromButton",
        [robot, stick, from_button],
        preconditions={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(RobotAboveButton, [robot, from_button]),
        },
        add_effects={
            LiftedAtom(Grasped, [robot, stick]),
            LiftedAtom(AboveNoButton, []),
        },
        delete_effects={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(RobotAboveButton, [robot, from_button]),
        },
    )

    # StickPressButtonFromNothing
    StickPressButtonFromNothingOperator = LiftedOperator(
        "StickPressButtonFromNothing",
        [robot, stick, button],
        preconditions={
            LiftedAtom(Grasped, [robot, stick]),
            LiftedAtom(AboveNoButton, []),
        },
        add_effects={
            LiftedAtom(StickAboveButton, [stick, button]),
            LiftedAtom(Pressed, [button]),
        },
        delete_effects={LiftedAtom(AboveNoButton, [])},
    )

    # StickPressButtonFromButton
    StickPressButtonFromButtonOperator = LiftedOperator(
        "StickPressButtonFromButton",
        [robot, stick, button, from_button],
        preconditions={
            LiftedAtom(Grasped, [robot, stick]),
            LiftedAtom(StickAboveButton, [stick, from_button]),
        },
        add_effects={
            LiftedAtom(StickAboveButton, [stick, button]),
            LiftedAtom(Pressed, [button]),
        },
        delete_effects={LiftedAtom(StickAboveButton, [stick, from_button])},
    )

    # PlaceStick
    PlaceStickOperator = LiftedOperator(
        "PlaceStick",
        [robot, stick],
        preconditions={
            LiftedAtom(Grasped, [robot, stick]),
        },
        add_effects={LiftedAtom(HandEmpty, [robot]), LiftedAtom(AboveNoButton, [])},
        delete_effects={LiftedAtom(Grasped, [robot, stick])},
    )

    # Controllers.
    class _CommonGroundController(GroundParameterizedController, abc.ABC):
        """Shared controller code between different actions."""

        def __init__(
            self,
            objects: Sequence[Object],
            safe_y: float = 0.8,
        ) -> None:
            self._robot = objects[0]
            assert self._robot.is_instance(CRVRobotType)
            super().__init__(objects)
            self._current_params: float = 0.0
            self._current_plan: list[NDArray[np.float32]] | None = None
            self._current_state: ObjectCentricState | None = None
            self._safe_y = safe_y
            # Extract max deltas from action space bounds
            assert isinstance(action_space, CRVRobotActionSpace)
            self._max_delta_x = action_space.high[0]
            self._max_delta_y = action_space.high[1]
            self._max_delta_theta = action_space.high[2]
            self._max_delta_arm = action_space.high[3]

        @abc.abstractmethod
        def _generate_waypoints(
            self, state: ObjectCentricState
        ) -> list[tuple[float, float, float, float]]:
            """Generate a waypoint plan with x, y, theta, arm values."""

        @abc.abstractmethod
        def _get_vacuum_actions(self) -> tuple[float, float]:
            """Get vacuum actions for during and after waypoint movement."""

        def _waypoints_to_plan(
            self,
            state: ObjectCentricState,
            waypoints: list[tuple[float, float, float, float]],
            vacuum_during_plan: float,
        ) -> list[NDArray[np.float32]]:
            current_pos = (
                state.get(self._robot, "x"),
                state.get(self._robot, "y"),
                state.get(self._robot, "theta"),
                state.get(self._robot, "arm_joint"),
            )
            waypoints = [current_pos] + waypoints
            plan: list[NDArray[np.float32]] = []
            for start, end in zip(waypoints[:-1], waypoints[1:]):
                if np.allclose(start, end):
                    continue
                total_dx = end[0] - start[0]
                total_dy = end[1] - start[1]
                total_dtheta = end[2] - start[2]
                total_darm = end[3] - start[3]
                num_steps = int(
                    max(
                        np.ceil(abs(total_dx) / self._max_delta_x),
                        np.ceil(abs(total_dy) / self._max_delta_y),
                        np.ceil(abs(total_dtheta) / self._max_delta_theta),
                        np.ceil(abs(total_darm) / self._max_delta_arm),
                    )
                )
                dx = total_dx / num_steps
                dy = total_dy / num_steps
                dtheta = total_dtheta / num_steps
                darm = total_darm / num_steps
                action = np.array(
                    [dx, dy, dtheta, darm, vacuum_during_plan], dtype=np.float32
                )
                for _ in range(num_steps):
                    plan.append(action)

            return plan

        def reset(self, x: ObjectCentricState, params: float) -> None:
            self._current_params = params
            self._current_plan = None
            self._current_state = x

        def terminated(self) -> bool:
            return self._current_plan is not None and len(self._current_plan) == 0

        def step(self) -> NDArray[np.float32]:
            assert self._current_state is not None
            if self._current_plan is None:
                self._current_plan = self._generate_plan(self._current_state)
            return self._current_plan.pop(0)

        def observe(self, x: ObjectCentricState) -> None:
            self._current_state = x

        def _generate_plan(self, x: ObjectCentricState) -> list[NDArray[np.float32]]:
            waypoints = self._generate_waypoints(x)
            vacuum_during_plan, vacuum_after_plan = self._get_vacuum_actions()
            waypoint_plan = self._waypoints_to_plan(x, waypoints, vacuum_during_plan)
            assert isinstance(action_space, CRVRobotActionSpace)
            plan_suffix: list[NDArray[np.float32]] = [
                # Change the vacuum.
                np.array([0, 0, 0, 0, vacuum_after_plan], dtype=np.float32),
            ]
            return waypoint_plan + plan_suffix

    class GroundPickStickController(GroundParameterizedController):
        """Controller for grasping the stick when the robot's hand is free."""

        def __init__(self, objects: Sequence[Object]) -> None:
            self._robot = objects[0]
            self._stick = objects[1]
            assert self._robot.is_instance(CRVRobotType)
            assert self._stick.is_instance(RectangleType)
            super().__init__(objects)
            self._current_params: tuple[float, float] = (0.0, 0.0)
            self._current_plan: list[NDArray[np.float32]] | None = None
            self._current_state: ObjectCentricState | None = None
            # Extract max deltas from action space bounds
            assert isinstance(action_space, CRVRobotActionSpace)
            self._max_delta_x = action_space.high[0]
            self._max_delta_y = action_space.high[1]
            self._max_delta_theta = action_space.high[2]
            self._max_delta_arm = action_space.high[3]

        def sample_parameters(
            self, x: ObjectCentricState, rng: np.random.Generator
        ) -> tuple[float, float]:
            # Sample grasp ratio along the stick [0,1] and desired arm length
            grasp_ratio = rng.uniform(0.0, 1.0)
            max_arm_length = x.get(self._robot, "arm_length")
            min_arm_length = x.get(self._robot, "base_radius")
            arm_length = rng.uniform(min_arm_length, max_arm_length)
            return (grasp_ratio, arm_length)

        def reset(self, x: ObjectCentricState, params: tuple[float, float]) -> None:
            self._current_params = params
            self._current_plan = None
            self._current_state = x

        def terminated(self) -> bool:
            return self._current_plan is not None and len(self._current_plan) == 0

        def step(self) -> NDArray[np.float32]:
            # Always extend the arm first before planning
            assert self._current_state is not None
            if self._current_plan is None:
                self._current_plan = self._generate_plan(self._current_state)
            return self._current_plan.pop(0)

        def observe(self, x: ObjectCentricState) -> None:
            self._current_state = x

        def _calculate_grasp_point(
            self, state: ObjectCentricState
        ) -> tuple[float, float]:
            """Calculate the actual grasp point based on ratio parameter."""
            grasp_ratio, _ = self._current_params

            # Get stick properties
            stick_x = state.get(self._stick, "x")
            stick_y = state.get(self._stick, "y")
            stick_width = state.get(self._stick, "width")

            # Get robot gripper properties
            gripper_height = state.get(self._robot, "gripper_height")

            full_line_length = stick_width + 2 * gripper_height
            line_length = full_line_length * grasp_ratio
            side_ratio = gripper_height / full_line_length
            bottom_ratio = stick_width / full_line_length

            # Define the grasping line from left bottom to right bottom of stick
            # Line starts at left edge and extends by gripper width on each side
            left_x = stick_x
            right_x = stick_x + stick_width
            bottom_y = stick_y
            grasp_x: float = 0.0
            grasp_y: float = 0.0

            if grasp_ratio < side_ratio:  # Grasping from left side
                grasp_x = left_x
                grasp_y = bottom_y + (gripper_height - line_length)
            elif (
                side_ratio <= grasp_ratio < side_ratio + bottom_ratio
            ):  # Grasping from bottom
                grasp_x = left_x + (grasp_ratio - side_ratio) * full_line_length
                grasp_y = bottom_y
            else:  # Grasping from right side
                grasp_x = right_x
                grasp_y = bottom_y + (line_length - gripper_height - stick_width)

            return grasp_x, grasp_y

        def _calculate_robot_position(
            self, state: ObjectCentricState, grasp_x: float, grasp_y: float
        ) -> tuple[float, float, float]:
            """Calculate robot position and orientation to reach grasp point."""
            _, desired_arm_length = self._current_params

            # Get stick properties
            stick_x = state.get(self._stick, "x")
            stick_width = state.get(self._stick, "width")

            # Get robot properties
            gripper_width = state.get(self._robot, "gripper_width")

            # Determine which side of the stick we're grasping from
            stick_left = stick_x
            stick_right = stick_x + stick_width

            robot_x: float = 0.0
            robot_y: float = 0.0
            robot_theta: float = 0.0

            if grasp_x < stick_left + stick_width * 0.01:  # Left side
                robot_x = grasp_x - desired_arm_length - gripper_width
                robot_y = grasp_y
                robot_theta = 0.0  # Facing right
            elif grasp_x > stick_right - stick_width * 0.01:  # Right side
                robot_x = grasp_x + desired_arm_length + gripper_width
                robot_y = grasp_y
                robot_theta = np.pi  # Facing left
            else:  # Bottom side
                robot_x = grasp_x
                robot_y = grasp_y - desired_arm_length - gripper_width
                robot_theta = np.pi / 2  # Facing up

            return robot_x, robot_y, robot_theta

        def _generate_waypoints(
            self, state: ObjectCentricState
        ) -> list[tuple[float, float, float, float]]:
            robot_x = state.get(self._robot, "x")
            robot_theta = state.get(self._robot, "theta")
            robot_radius = state.get(self._robot, "base_radius")
            robot_gripper_width = state.get(self._robot, "gripper_width")
            safe_y = robot_radius + robot_gripper_width * 2

            # Calculate grasp point and robot target position
            grasp_x, grasp_y = self._calculate_grasp_point(state)
            target_x, target_y, target_theta = self._calculate_robot_position(
                state, grasp_x, grasp_y
            )
            _, desired_arm_length = self._current_params

            return [
                # Start by moving the arm inside the robot's base
                (robot_x, safe_y, robot_theta, robot_radius),
                # Start by moving to safe height with current orientation
                (robot_x, safe_y, robot_theta, robot_radius),
                # Move to target x position at safe height
                (target_x, safe_y, robot_theta, robot_radius),
                # Orient towards the stick
                (target_x, safe_y, target_theta, robot_radius),
                # Move down to grasp position
                (target_x, target_y, target_theta, robot_radius),
                # Extend arm to desired length
                (target_x, target_y, target_theta, desired_arm_length),
            ]

        def _waypoints_to_plan(
            self,
            state: ObjectCentricState,
            waypoints: list[tuple[float, float, float, float]],
            vacuum_during_plan: float,
        ) -> list[NDArray[np.float32]]:
            current_pos = (
                state.get(self._robot, "x"),
                state.get(self._robot, "y"),
                state.get(self._robot, "theta"),
                state.get(self._robot, "arm_joint"),
            )
            waypoints = [current_pos] + waypoints
            plan: list[NDArray[np.float32]] = []
            for start, end in zip(waypoints[:-1], waypoints[1:]):
                if np.allclose(start, end):
                    continue
                total_dx = end[0] - start[0]
                total_dy = end[1] - start[1]
                total_dtheta = end[2] - start[2]
                total_darm = end[3] - start[3]
                num_steps = int(
                    max(
                        np.ceil(abs(total_dx) / self._max_delta_x),
                        np.ceil(abs(total_dy) / self._max_delta_y),
                        np.ceil(abs(total_dtheta) / self._max_delta_theta),
                        np.ceil(abs(total_darm) / self._max_delta_arm),
                    )
                )
                dx = total_dx / num_steps
                dy = total_dy / num_steps
                dtheta = total_dtheta / num_steps
                darm = total_darm / num_steps
                action = np.array(
                    [dx, dy, dtheta, darm, vacuum_during_plan], dtype=np.float32
                )
                for _ in range(num_steps):
                    plan.append(action)

            return plan

        def _generate_plan(self, x: ObjectCentricState) -> list[NDArray[np.float32]]:
            waypoints = self._generate_waypoints(x)
            vacuum_during_plan = 0.0  # Vacuum OFF during movement
            vacuum_after_plan = 1.0  # Vacuum ON after reaching grasp position
            waypoint_plan = self._waypoints_to_plan(x, waypoints, vacuum_during_plan)
            assert isinstance(action_space, CRVRobotActionSpace)
            plan_suffix: list[NDArray[np.float32]] = [
                # Change the vacuum to grasp
                np.array([0, 0, 0, 0, vacuum_after_plan], dtype=np.float32),
            ]
            return waypoint_plan + plan_suffix

    class GroundPlaceStickController(_CommonGroundController):
        """Controller for releasing the stick."""

        def __init__(self, objects: Sequence[Object], **kwargs) -> None:
            super().__init__(objects, **kwargs)
            self._stick = objects[1]
            assert self._stick.is_instance(RectangleType)

        def sample_parameters(
            self, x: ObjectCentricState, rng: np.random.Generator
        ) -> float:
            # Parameter represents absolute x position where to release the stick
            del x, rng  # not used in this controller
            return 0.0

        def _generate_waypoints(
            self, state: ObjectCentricState
        ) -> list[tuple[float, float, float, float]]:
            robot_x = state.get(self._robot, "x")
            robot_y = state.get(self._robot, "y")
            robot_base_radius = state.get(self._robot, "base_radius")
            robot_theta = state.get(self._robot, "theta")

            return [
                # Just move the arm back
                (robot_x, robot_y, robot_theta, robot_base_radius),
            ]

        def _get_vacuum_actions(self) -> tuple[float, float]:
            return 0.0, 0.0

    class GroundRobotPressButtonController(_CommonGroundController):
        """Controller for pressing a button directly with the robot."""

        def __init__(self, objects: Sequence[Object], **kwargs) -> None:
            super().__init__(objects, **kwargs)
            self._button = objects[1]
            assert self._button.is_instance(CircleType)

        def sample_parameters(
            self, x: ObjectCentricState, rng: np.random.Generator
        ) -> float:
            del x, rng  # not used in this controller
            return 0.0

        def _generate_waypoints(
            self, state: ObjectCentricState
        ) -> list[tuple[float, float, float, float]]:
            robot_theta = state.get(self._robot, "theta")
            robot_radius = state.get(self._robot, "base_radius")
            button_x = state.get(self._button, "x")
            button_y = state.get(self._button, "y")

            # Position robot base to intersect with button
            # For intersection, robot base should be close to button center
            target_x = button_x
            target_y = button_y  # Put robot base at button level for intersection

            return [
                # Move down so robot base overlaps with button
                (target_x, target_y, robot_theta, robot_radius),
            ]

        def _get_vacuum_actions(self) -> tuple[float, float]:
            return 0.0, 0.0

    class GroundStickPressButtonController(_CommonGroundController):
        """Controller for pressing a button using the stick."""

        def __init__(self, objects: Sequence[Object], **kwargs) -> None:
            super().__init__(objects, **kwargs)
            self._stick = objects[1]
            self._button = objects[2]
            assert self._stick.is_instance(RectangleType)
            assert self._button.is_instance(CircleType)

        def sample_parameters(
            self, x: ObjectCentricState, rng: np.random.Generator
        ) -> float:
            del x, rng  # not used in this controller
            return 0.0

        def _generate_waypoints(
            self, state: ObjectCentricState
        ) -> list[tuple[float, float, float, float]]:
            """Assume we always use the stick far end to press the button."""
            robot_x = state.get(self._robot, "x")
            robot_y = state.get(self._robot, "y")
            robot_theta = state.get(self._robot, "theta")
            robot_arm_joint = state.get(self._robot, "arm_joint")
            button_x = state.get(self._button, "x")
            button_y = state.get(self._button, "y")
            stick_far_x = state.get(self._stick, "x") + state.get(self._stick, "width")
            stick_far_y = state.get(self._stick, "y") + state.get(self._stick, "height")

            dx = button_x - stick_far_x
            dy = button_y - stick_far_y

            # Position robot so stick can reach button
            # Account for stick length and robot arm extension
            target_x = robot_x + dx
            target_y = robot_y + dy

            return [
                # Move down to press button with stick
                (target_x, target_y, robot_theta, robot_arm_joint),
            ]

        def _get_vacuum_actions(self) -> tuple[float, float]:
            return 1.0, 1.0  # Keep holding stick

    # Create lifted controllers that match operator parameters exactly

    RobotPressButtonFromNothingController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, button],
            GroundRobotPressButtonController,
        )
    )

    RobotPressButtonFromButtonController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, button, from_button],
            GroundRobotPressButtonController,
        )
    )

    PickStickFromNothingController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, stick],
            GroundPickStickController,
        )
    )

    PickStickFromButtonController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, stick, from_button],
            GroundPickStickController,
        )
    )

    StickPressButtonFromNothingController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, stick, button],
            GroundStickPressButtonController,
        )
    )

    StickPressButtonFromButtonController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, stick, button, from_button],
            GroundStickPressButtonController,
        )
    )

    PlaceStickController: LiftedParameterizedController = LiftedParameterizedController(
        [robot, stick],
        GroundPlaceStickController,
    )

    # Finalize the skills.
    skills = {
        LiftedSkill(PickStickFromNothingOperator, PickStickFromNothingController),
        LiftedSkill(PickStickFromButtonOperator, PickStickFromButtonController),
        LiftedSkill(PlaceStickOperator, PlaceStickController),
        LiftedSkill(
            RobotPressButtonFromNothingOperator, RobotPressButtonFromNothingController
        ),
        LiftedSkill(
            RobotPressButtonFromButtonOperator, RobotPressButtonFromButtonController
        ),
        LiftedSkill(
            StickPressButtonFromNothingOperator, StickPressButtonFromNothingController
        ),
        LiftedSkill(
            StickPressButtonFromButtonOperator, StickPressButtonFromButtonController
        ),
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
