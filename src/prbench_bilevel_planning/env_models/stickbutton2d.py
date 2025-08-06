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
)
from geom2drobotenvs.object_types import CircleType, CRVRobotType, RectangleType
from geom2drobotenvs.utils import (
    CRVRobotActionSpace,
    get_suctioned_objects,
)
from gymnasium.spaces import Box, Space
from numpy.typing import NDArray
from prbench.envs.stickbutton2d import (
    ObjectCentricStickButton2DEnv,
    StickButton2DEnvSpec,
)
from prpl_utils.spaces import FunctionalSpace
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
from tomsgeoms2d.structs import Circle, Geom2D, Rectangle

from prbench_bilevel_planning.structs import BilevelPlanningEnvModels


def object_to_geom(obj: Object, state: ObjectCentricState) -> Geom2D:
    """Convert an object to its geometric representation."""
    if obj.is_instance(CRVRobotType):
        # Robot is a circle
        x = state.get(obj, "x")
        y = state.get(obj, "y")
        radius = state.get(obj, "base_radius")
        return Circle(x, y, radius)
    elif obj.is_instance(RectangleType):
        # Stick is a rectangle
        x = state.get(obj, "x")
        y = state.get(obj, "y")
        theta = state.get(obj, "theta")
        width = state.get(obj, "width")
        height = state.get(obj, "height")
        return Rectangle(x, y, theta, width, height)
    elif obj.is_instance(CircleType):
        # Button is a circle
        x = state.get(obj, "x")
        y = state.get(obj, "y")
        radius = state.get(obj, "radius")
        return Circle(x, y, radius)
    else:
        raise ValueError(f"Unknown object type: {obj.type}")


def create_bilevel_planning_models(
    observation_space: Space, executable_space: Space, num_buttons: int
) -> BilevelPlanningEnvModels:
    """Create the env models for stick button 2D."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(executable_space, CRVRobotActionSpace)

    # Make a local copy of the environment to use as the "simulator". Note that we use
    # the object-centric version of the environment because we want access to the reset
    # and step functions in there, which operate over ObjectCentricState, which we use
    # as the state representation for planning.
    sim = ObjectCentricStickButton2DEnv(num_buttons=num_buttons)

    # Convert observations into states. The important thing is that states are hashable.
    def observation_to_state(o: NDArray[np.float32]) -> ObjectCentricState:
        """Convert the vectors back into (hashable) object-centric states."""
        return observation_space.devectorize(o)

    # Convert actions into executable actions. Actions must be hashable.
    def action_to_executable(action: tuple[float, ...]) -> NDArray[np.float32]:
        """Convert actions into executables."""
        return np.array(action, dtype=np.float32)

    # The object-centric states that are passed around in planning do not include the
    # globally constant objects, so we need to create an exemplar state that does
    # include them and then copy in the changing values before calling step().
    exemplar_state = sim.reset()[0]

    # Create the transition function.
    def transition_fn(
        x: ObjectCentricState, u: tuple[float, ...]
    ) -> ObjectCentricState:
        """Simulate the action."""
        # See note above re: why we can't just sim.reset(options={"init_state": x}).
        state = exemplar_state.copy()
        for obj, feats in x.data.items():
            state.data[obj] = feats
        # Now we can reset().
        sim.reset(options={"init_state": state})
        sim_obs, _, _, _, _ = sim.step(action_to_executable(u))

        # Now we need to extract back out the changing objects.
        next_x = x.copy()
        for obj in x:
            next_x.data[obj] = sim_obs.data[obj]
        return next_x

    # Types.
    types = {CRVRobotType, RectangleType, CircleType}

    # Create the state space.
    state_space = ObjectCentricStateSpace(types)

    # Create the action space.
    action_space: Space = FunctionalSpace(
        contains_fn=lambda x: isinstance(x, tuple)
    )  # weak

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
        robot = Object("robot", CRVRobotType)
        stick = Object("stick", RectangleType)
        buttons: set[Object] = set()
        for i in range(num_buttons):
            button = Object(f"button{i}", CircleType)
            buttons.add(button)

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
        robot_geom = object_to_geom(robot, x)
        stick_geom = object_to_geom(stick, x)

        robot_above_any_button = False
        stick_above_any_button = False

        for button in buttons:
            button_geom = object_to_geom(button, x)

            # Check if robot is above button (geometrically intersects)
            if robot_geom.intersects(button_geom):
                atoms.add(GroundAtom(RobotAboveButton, [robot, button]))
                robot_above_any_button = True

            # Check if stick is above button (geometrically intersects)
            if stick_geom.intersects(button_geom):
                atoms.add(GroundAtom(StickAboveButton, [stick, button]))
                stick_above_any_button = True

        # Add AboveNoButton if neither robot nor stick is above any button
        if not robot_above_any_button and not stick_above_any_button:
            atoms.add(GroundAtom(AboveNoButton, []))

        objects = {robot, stick} | buttons
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
        },
        delete_effects={
            LiftedAtom(AboveNoButton, [])
        }
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
            max_delta: float = 0.025,
        ) -> None:
            self._robot = objects[0]
            assert self._robot.is_instance(CRVRobotType)
            super().__init__(objects)
            self._current_params: float = 0.0
            self._current_plan: list[tuple[float, ...]] | None = None
            self._current_state: ObjectCentricState | None = None
            self._max_delta = max_delta

        @abc.abstractmethod
        def _generate_waypoints(
            self, state: ObjectCentricState
        ) -> list[tuple[float, float]]:
            """Generate a waypoint plan."""

        @abc.abstractmethod
        def _get_vacuum_actions(self) -> tuple[float, float]:
            """Get vacuum actions for during and after waypoint movement."""

        def _waypoints_to_plan(
            self,
            state: ObjectCentricState,
            waypoints: list[tuple[float, float]],
            vacuum_during_plan: float,
        ) -> list[tuple[float, ...]]:
            current_pos = (state.get(self._robot, "x"), state.get(self._robot, "y"))
            waypoints = [current_pos] + waypoints
            plan: list[tuple[float, ...]] = []
            for start, end in zip(waypoints[:-1], waypoints[1:]):
                if np.allclose(start, end):
                    continue
                total_dx = end[0] - start[0]
                total_dy = end[1] - start[1]
                num_steps = int(
                    max(
                        np.ceil(abs(total_dx) / self._max_delta),
                        np.ceil(abs(total_dy) / self._max_delta),
                    )
                )
                dx = total_dx / num_steps
                dy = total_dy / num_steps
                action = (dx, dy, 0, 0, vacuum_during_plan)
                for _ in range(num_steps):
                    plan.append(action)

            return plan

        def reset(self, x: ObjectCentricState, params: float) -> None:
            self._current_params = params
            self._current_plan = None
            self._current_state = x

        def terminated(self) -> bool:
            return self._current_plan is not None and len(self._current_plan) == 0

        def step(self) -> tuple[float, ...]:
            # Always extend the arm first before planning (same as obstruction2d)
            assert self._current_state is not None
            if self._current_state.get(self._robot, "arm_joint") <= 0.15:
                assert isinstance(executable_space, Box)
                return (0, 0, 0, executable_space.high[3], 0)
            if self._current_plan is None:
                self._current_plan = self._generate_plan(self._current_state)
            return self._current_plan.pop(0)

        def observe(self, x: ObjectCentricState) -> None:
            self._current_state = x

        def _generate_plan(self, x: ObjectCentricState) -> list[tuple[float, ...]]:
            waypoints = self._generate_waypoints(x)
            vacuum_during_plan, vacuum_after_plan = self._get_vacuum_actions()
            waypoint_plan = self._waypoints_to_plan(x, waypoints, vacuum_during_plan)
            assert isinstance(executable_space, Box)
            plan_suffix: list[tuple[float, ...]] = [
                # Change the vacuum.
                (0, 0, 0, 0, vacuum_after_plan),
                # Move up slightly to break contact.
                (0, executable_space.high[1], 0, 0, vacuum_after_plan),
            ]
            return waypoint_plan + plan_suffix

    class GroundPickStickController(_CommonGroundController):
        """Controller for grasping the stick when the robot's hand is free."""

        def __init__(self, objects: Sequence[Object], **kwargs) -> None:
            super().__init__(objects, **kwargs)
            self._stick = objects[1]
            assert self._stick.is_instance(RectangleType)

        def sample_parameters(
            self, x: ObjectCentricState, rng: np.random.Generator
        ) -> float:
            # Use a much simpler approach - try to get as close as possible
            # The obstruction2d approach works by sampling different positions
            gripper_height = x.get(self._robot, "gripper_height")
            stick_width = x.get(self._stick, "width")
            # Sample positions around the stick with small offsets
            params = rng.uniform(-gripper_height / 2, stick_width + gripper_height / 2)
            return params

        def _generate_waypoints(
            self, state: ObjectCentricState
        ) -> list[tuple[float, float]]:
            robot_x = state.get(self._robot, "x")
            stick_x = state.get(self._stick, "x")
            robot_arm_joint = state.get(self._robot, "arm_joint")
            target_x, target_y = get_robot_stick_grasp_position(
                self._stick,
                state,
                stick_x,
                robot_arm_joint,
                relative_x_offset=self._current_params,
            )
            return [
                # Start by moving to safe height
                (robot_x, self._safe_y),
                # Move to above the stick, offset by params
                (target_x, self._safe_y),
                # Move down to grasp
                (target_x, target_y),
            ]

        def _get_vacuum_actions(self) -> tuple[float, float]:
            return 0.0, 1.0

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
            world_min_x = 0.5
            world_max_x = 3.0
            return rng.uniform(world_min_x, world_max_x)

        def _generate_waypoints(
            self, state: ObjectCentricState
        ) -> list[tuple[float, float]]:
            robot_x = state.get(self._robot, "x")
            robot_arm_joint = state.get(self._robot, "arm_joint")

            # Move to release position
            release_x = self._current_params
            release_y = 1.5  # Place on table level
            target_y = (
                release_y
                + robot_arm_joint
                + state.get(self._robot, "gripper_width") / 2
            )

            return [
                # Start by moving to safe height
                (robot_x, self._safe_y),
                # Move to release position
                (release_x, self._safe_y),
                # Move down to place
                (release_x, target_y),
            ]

        def _get_vacuum_actions(self) -> tuple[float, float]:
            return 1.0, 0.0

    class GroundRobotPressButtonController(_CommonGroundController):
        """Controller for pressing a button directly with the robot."""

        def __init__(self, objects: Sequence[Object], **kwargs) -> None:
            super().__init__(objects, **kwargs)
            self._button = objects[1]
            assert self._button.is_instance(CircleType)

        def sample_parameters(
            self, x: ObjectCentricState, rng: np.random.Generator
        ) -> float:
            # Parameter represents approach angle offset
            return rng.uniform(-0.1, 0.1)

        def _generate_waypoints(
            self, state: ObjectCentricState
        ) -> list[tuple[float, float]]:
            button_x = state.get(self._button, "x")
            button_y = state.get(self._button, "y")
            robot_arm_joint = state.get(self._robot, "arm_joint")

            # Calculate approach position
            target_x = button_x + self._current_params
            target_y = button_y + robot_arm_joint

            return [
                # Move to touch button
                (target_x, target_y),
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
            # Parameter represents positioning strategy
            return rng.uniform(-0.2, 0.2)

        def _generate_waypoints(
            self, state: ObjectCentricState
        ) -> list[tuple[float, float]]:
            robot_x = state.get(self._robot, "x")
            button_x = state.get(self._button, "x")
            button_y = state.get(self._button, "y")
            stick_height = state.get(self._stick, "height")
            robot_arm_joint = state.get(self._robot, "arm_joint")

            # Position robot so stick can reach button
            # Account for stick length and robot arm extension
            target_x = button_x - stick_height / 2 + self._current_params
            target_y = button_y + robot_arm_joint

            return []

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
        LiftedSkill(RobotPressButtonFromNothingOperator, RobotPressButtonFromNothingController),
        LiftedSkill(RobotPressButtonFromButtonOperator, RobotPressButtonFromButtonController),
        LiftedSkill(StickPressButtonFromNothingOperator, StickPressButtonFromNothingController),
        LiftedSkill(StickPressButtonFromButtonOperator, StickPressButtonFromButtonController),
    }

    # Finalize the models.
    return BilevelPlanningEnvModels(
        observation_space,
        executable_space,
        state_space,
        action_space,
        transition_fn,
        types,
        predicates,
        observation_to_state,
        action_to_executable,
        state_abstractor,
        goal_deriver,
        skills,
    )
