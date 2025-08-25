#!/usr/bin/env python3
"""Example usage of Tidybot 3D pick-and-place skills with bilevel planning.

This script demonstrates how to use the implemented tidybot pick-and-place skills with
the bilevel planning framework, integrating with the tidybot MuJoCo environment and
controllers.
"""

import time
from typing import Any, Dict, Optional

import numpy as np

# Import bilevel planning components
try:
    from bilevel_planning.structs import SesameModels
    from relational_structs import Object, ObjectCentricState, Type
    from relational_structs.utils import create_state_from_dict
except ImportError:
    print("Warning: Bilevel planning components not available")
    SesameModels = None
    Object = None
    Type = None
    ObjectCentricState = None
    create_state_from_dict = None

# Import tidybot components
try:
    from prbench.envs.tidybot.tidybot_mujoco_env import MujocoEnv
except ImportError:
    print("Warning: Tidybot components not available")
    MujocoEnv = None

# Import our implemented skills
from prbench_bilevel_planning.env_models.tidybot3d import (
    PickController,
    PlaceController,
    TidybotStateConverter,
    create_tidybot_action,
)


class TidybotBilevelDemo:
    """Demonstration of tidybot bilevel planning integration."""

    def __init__(self, use_mujoco: bool = True, show_viewer: bool = True):
        """Initialize the demo environment.

        Args:
            use_mujoco: Whether to use the MuJoCo environment
            show_viewer: Whether to show the MuJoCo viewer
        """
        self.use_mujoco = use_mujoco
        self.show_viewer = show_viewer
        self.env: Optional[MujocoEnv] = None

        # Initialize environment if available
        if use_mujoco and MujocoEnv is not None:
            self._setup_mujoco_env()

        # Create example objects and state
        self._setup_example_state()

    def _state_as_dict(self) -> dict:
        """Reconstruct a dict[Object, dict[str, float]] from current state."""
        state_dict: dict = {}
        for obj in self.state.data.keys():
            # Get feature list for this object's type
            feats = self.type_features.get(obj.type, [])
            obj_dict: dict[str, float] = {}
            for feat in feats:
                obj_dict[feat] = float(self.state.get(obj, feat))
            state_dict[obj] = obj_dict
        return state_dict

    def _setup_mujoco_env(self) -> None:
        """Set up the MuJoCo environment."""
        print("Setting up MuJoCo environment...")
        try:
            self.env = MujocoEnv(
                render_images=True,
                show_viewer=self.show_viewer,
                table_scene=False,  # Use ground scene
                cupboard_scene=False,
            )

            print("MuJoCo environment setup complete")
        except Exception as e:
            print(f"Failed to setup MuJoCo environment: {e}")
            self.env = None

    def _setup_example_state(self) -> None:
        """Set up example objects and state for demonstration."""
        if Type is None or Object is None or create_state_from_dict is None:
            print("Warning: Cannot create example state without relational_structs")
            self.robot = None
            self.cube = None
            self.state = None
            return

        # Define object types
        robot_type = Type(
            "robot", ["x", "y", "theta", "arm_x", "arm_y", "arm_z", "gripper"]
        )
        cube_type = Type("cube", ["x", "y", "z"])

        # Create objects
        self.robot = Object("tidybot", robot_type)
        self.cube = Object("cube1", cube_type)

        # Create initial state dictionary
        initial_state_data = {
            self.robot: {
                "x": 0.0,
                "y": 0.0,
                "theta": 0.0,
                "arm_x": 0.14,
                "arm_y": 0.0,
                "arm_z": 0.21,
                "gripper": 0.0,
            },
            self.cube: {
                "x": 0.5,
                "y": 0.1,
                "z": 0.02,
            },
        }

        # Create type features dictionary and store for updates
        self.type_features = {
            robot_type: ["x", "y", "theta", "arm_x", "arm_y", "arm_z", "gripper"],
            cube_type: ["x", "y", "z"],
        }

        # Create state using the proper function
        self.state = create_state_from_dict(initial_state_data, self.type_features)
        print("Example state created with robot at origin and cube at (0.5, 0.1)")

    def demonstrate_pick_skill(self) -> None:
        """Demonstrate the pick skill."""
        print("\n=== Demonstrating Pick Skill ===")

        if self.robot is None or self.cube is None or self.state is None:
            print("Cannot demonstrate without proper state setup")
            return

        # Create pick controller
        pick_controller = PickController(
            objects=[self.robot, self.cube],
            max_skill_horizon=100,
            custom_grasp=False,
            env=self.env,  # Pass the MuJoCo environment
        )

        # Reset controller with current state
        pick_controller.reset(self.state, params=0.0)

        print(
            f"Initial robot position: {self.state.get(self.robot, 'x'):.2f}, {self.state.get(self.robot, 'y'):.2f}"
        )
        print(
            f"Target cube position: {self.state.get(self.cube, 'x'):.2f}, {self.state.get(self.cube, 'y'):.2f}"
        )

        # Execute skill steps
        step_count = 0
        while not pick_controller.terminated() and step_count < 100:
            action = pick_controller.step()

            # Update state based on action (now uses MuJoCo state)
            self._update_state_from_action(action)
            pick_controller.observe(self.state)

            # Print current robot position and action
            current_robot_pos = (
                self.state.get(self.robot, "x"),
                self.state.get(self.robot, "y"),
            )
            print(
                f"Step {step_count}: Robot at ({current_robot_pos[0]:.3f}, {current_robot_pos[1]:.3f}), Action base_pose={action['base_pose']}"
            )

            step_count += 1

            # If using MuJoCo, execute action in environment
            if self.env is not None:
                self.env.step(action)
                time.sleep(0.1)  # Small delay for visualization

        print(f"Pick skill completed in {step_count} steps")

    def demonstrate_place_skill(self) -> None:
        """Demonstrate the place skill."""
        print("\n=== Demonstrating Place Skill ===")

        if self.robot is None or self.cube is None or self.state is None:
            print("Cannot demonstrate without proper state setup")
            return

        # Create place controller with target location
        target_x, target_y, target_z = 0.0, 0.0, 0.5
        place_controller = PlaceController(
            objects=[self.robot, self.cube],
            target_x=target_x,
            target_y=target_y,
            target_z=target_z,
            max_skill_horizon=100,
            env=self.env,  # Pass the MuJoCo environment
        )

        # Reset controller with current state
        place_controller.reset(self.state, params=0.0)

        print(
            f"Current robot position: {self.state.get(self.robot, 'x'):.2f}, {self.state.get(self.robot, 'y'):.2f}"
        )
        print(f"Target place location: {target_x:.2f}, {target_y:.2f}")

        # Execute skill steps
        step_count = 0
        while not place_controller.terminated() and step_count < 100:
            action = place_controller.step()

            # Update state based on action (now uses MuJoCo state)
            self._update_state_from_action(action)
            place_controller.observe(self.state)

            # Print current robot position and action
            current_robot_pos = (
                self.state.get(self.robot, "x"),
                self.state.get(self.robot, "y"),
            )
            print(
                f"Step {step_count}: Robot at ({current_robot_pos[0]:.3f}, {current_robot_pos[1]:.3f}), Action base_pose={action['base_pose']}"
            )

            step_count += 1

            # If using MuJoCo, execute action in environment
            if self.env is not None:
                self.env.step(action)
                time.sleep(0.1)

        print(f"Place skill completed in {step_count} steps")

    def _update_state_from_mujoco(self) -> None:
        """Update the state from the actual MuJoCo environment."""
        if self.env is None or self.robot is None:
            return

        # Get current observation from MuJoCo
        obs = self.env.get_obs()

        # Update robot state from MuJoCo observation
        if "base_pose" in obs:
            base_pose = obs["base_pose"]
            robot_updates = {
                "x": float(base_pose[0]),
                "y": float(base_pose[1]),
                "theta": float(base_pose[2]),
            }
        else:
            robot_updates = {}

        if "arm_pos" in obs:
            arm_pos = obs["arm_pos"]
            robot_updates.update(
                {
                    "arm_x": float(arm_pos[0]),
                    "arm_y": float(arm_pos[1]),
                    "arm_z": float(arm_pos[2]),
                }
            )

        if "gripper_pos" in obs:
            gripper_pos = obs["gripper_pos"]
            robot_updates["gripper"] = float(gripper_pos[0])

        # Update cube positions from MuJoCo
        cube_updates = {}
        for key, value in obs.items():
            if key.endswith("_pos") and "cube" in key:
                cube_name = key.replace("_pos", "")
                if (
                    cube_name == "cube1" and self.cube is not None
                ):  # Update our tracked cube
                    cube_updates = {
                        "x": float(value[0]),
                        "y": float(value[1]),
                        "z": float(value[2]),
                    }
                    break

        # Rebuild the state with updated values
        new_state_dict = self._state_as_dict()
        if robot_updates:
            new_state_dict[self.robot] = {**new_state_dict[self.robot], **robot_updates}
        if cube_updates and self.cube is not None:
            new_state_dict[self.cube] = {**new_state_dict[self.cube], **cube_updates}

        self.state = create_state_from_dict(new_state_dict, self.type_features)

    def _update_state_from_action(self, action: Dict[str, Any]) -> None:
        """Update the state based on the executed action (simplified simulation)."""
        if self.env is not None:
            # If using MuJoCo, get actual state from environment instead of simulation
            self._update_state_from_mujoco()
            return

        # Fallback to simplified simulation if no MuJoCo environment
        if self.state is None or self.robot is None:
            return

        # Update robot position based on base_pose action
        if "base_pose" in action:
            base_pose = action["base_pose"]
            # Simple integration - move towards target pose
            current_x = self.state.get(self.robot, "x")
            current_y = self.state.get(self.robot, "y")
            current_theta = self.state.get(self.robot, "theta")

            # Simple proportional control towards target
            alpha = 0.1  # Learning rate
            new_x = float(current_x + alpha * (base_pose[0] - current_x))
            new_y = float(current_y + alpha * (base_pose[1] - current_y))
            new_theta = float(current_theta + alpha * (base_pose[2] - current_theta))

            # Build new state dict for robot and replace via create_state_from_dict
            new_robot_state = {
                "x": new_x,
                "y": new_y,
                "theta": new_theta,
                "arm_x": float(self.state.get(self.robot, "arm_x")),
                "arm_y": float(self.state.get(self.robot, "arm_y")),
                "arm_z": float(self.state.get(self.robot, "arm_z")),
                "gripper": float(self.state.get(self.robot, "gripper")),
            }
            new_state_dict = self._state_as_dict()
            new_state_dict[self.robot] = new_robot_state
            self.state = create_state_from_dict(new_state_dict, self.type_features)

        # Update arm position
        if "arm_pos" in action:
            arm_pos = action["arm_pos"]
            new_robot_state = {
                "x": float(self.state.get(self.robot, "x")),
                "y": float(self.state.get(self.robot, "y")),
                "theta": float(self.state.get(self.robot, "theta")),
                "arm_x": float(arm_pos[0]),
                "arm_y": float(arm_pos[1]),
                "arm_z": float(arm_pos[2]),
                "gripper": float(self.state.get(self.robot, "gripper")),
            }
            new_state_dict = self._state_as_dict()
            new_state_dict[self.robot] = new_robot_state
            self.state = create_state_from_dict(new_state_dict, self.type_features)

        # Update gripper
        if "gripper_pos" in action:
            gripper_pos = action["gripper_pos"]
            new_robot_state = {
                "x": float(self.state.get(self.robot, "x")),
                "y": float(self.state.get(self.robot, "y")),
                "theta": float(self.state.get(self.robot, "theta")),
                "arm_x": float(self.state.get(self.robot, "arm_x")),
                "arm_y": float(self.state.get(self.robot, "arm_y")),
                "arm_z": float(self.state.get(self.robot, "arm_z")),
                "gripper": float(gripper_pos[0]),
            }
            new_state_dict = self._state_as_dict()
            new_state_dict[self.robot] = new_robot_state
            self.state = create_state_from_dict(new_state_dict, self.type_features)

    def demonstrate_state_conversion(self) -> None:
        """Demonstrate state conversion utilities."""
        print("\n=== Demonstrating State Conversion ===")

        if self.state is None or self.robot is None:
            print("Cannot demonstrate without proper state setup")
            return

        # Convert state to observation
        obs = TidybotStateConverter.state_to_obs(self.state, self.robot)
        print("Converted state to observation:")
        for key, value in obs.items():
            print(f"  {key}: {value}")

        # Convert observation back to state updates
        updates = TidybotStateConverter.obs_to_state_update(obs, self.robot)
        print("\nConverted observation to state updates:")
        for obj, obj_updates in updates.items():
            print(f"  {obj.name}: {obj_updates}")

        # Demonstrate action creation
        action = create_tidybot_action(
            base_pose=np.array([0.5, 0.1, 0.0]),
            arm_pos=np.array([0.2, 0.0, 0.3]),
            gripper_pos=np.array([1.0]),
        )
        print("\nCreated tidybot action:")
        for key, value in action.items():
            print(f"  {key}: {value}")

    def run_full_demonstration(self) -> None:
        """Run the complete demonstration."""
        print("Starting Tidybot 3D Pick-and-Place Skills Demonstration")
        print("=" * 60)

        # Reset environment if using MuJoCo
        if self.env is not None:
            print("Resetting MuJoCo environment...")
            self.env.reset()
            time.sleep(1.0)  # Allow environment to stabilize

        # Demonstrate utilities
        self.demonstrate_state_conversion()

        # Demonstrate individual skills
        self.demonstrate_pick_skill()
        self.demonstrate_place_skill()

        # # Demonstrate combined skill
        # self.demonstrate_pick_and_place_skill()

        print("\n" + "=" * 60)
        print("Demonstration completed successfully!")

        # Cleanup
        if self.env is not None:
            print("Closing MuJoCo environment...")
            self.env.close()


def main():
    """Main function to run the demonstration."""
    print("Tidybot 3D Pick-and-Place Skills Demo")
    print("====================================")

    # Check if MuJoCo components are available
    use_mujoco = MujocoEnv is not None
    if not use_mujoco:
        print("MuJoCo components not available - running without visualization")

    # Create and run demonstration
    demo = TidybotBilevelDemo(use_mujoco=use_mujoco, show_viewer=True)

    try:
        demo.run_full_demonstration()
    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user")
    except Exception as e:
        print(f"\nDemonstration failed with error: {e}")
        import traceback

        traceback.print_exc()

    print("Demo finished")


if __name__ == "__main__":
    main()
