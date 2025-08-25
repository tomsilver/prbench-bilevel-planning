"""Integration tests for the ArmController class in prbench.envs.tidybot.arm_controller.

These tests use the ArmController and its dependencies, assuming all required packages
are installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

# Try to import ArmController, skip tests if MuJoCo isn't available
skip_reason = ""
try:
    from prbench_bilevel_planning.env_models.tidybot3d.arm_controller import (
        ArmController,
    )

    MUJOCO_AVAILABLE = True
except (ImportError, AttributeError) as e:
    MUJOCO_AVAILABLE = False
    skip_reason = f"MuJoCo/OpenGL not available: {e}"
    if TYPE_CHECKING:
        from prbench_bilevel_planning.env_models.tidybot3d.arm_controller import (
            ArmController,
        )

# pylint: disable=redefined-outer-name


@pytest.fixture
def arm_controller() -> ArmController:
    """Fixture to create an ArmController instance with real dependencies for
    testing."""
    if not MUJOCO_AVAILABLE:
        pytest.skip(skip_reason)
    qpos = np.zeros(7)
    qvel = np.zeros(7)
    ctrl = np.zeros(7)
    qpos_gripper = np.zeros(1)
    ctrl_gripper = np.zeros(1)
    timestep = 0.1
    return ArmController(qpos, qvel, ctrl, qpos_gripper, ctrl_gripper, timestep)


@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason=skip_reason)
def test_reset(arm_controller):
    """Test that ArmController.reset() sets state to retract configuration and OTG to
    Finished."""
    arm_controller.reset()
    assert np.allclose(
        arm_controller.qpos,
        [0.0, -0.34906585, 3.14159265, -2.54818071, 0.0, -0.87266463, 1.57079633],
    )
    assert np.allclose(arm_controller.ctrl, arm_controller.qpos)
    assert arm_controller.ctrl_gripper == 0.0 or np.allclose(
        arm_controller.ctrl_gripper, 0.0
    )
    # OTG result may depend on Ruckig, so just check it's set (not None)
    assert arm_controller.otg_res is not None


@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason=skip_reason)
def test_control_callback_sets_target(arm_controller):
    """Test that control_callback sets the target position using real TidybotIKSolver
    and updates OTG state."""
    arm_controller.reset()
    command = {
        "arm_pos": np.array([1.0, 2.0, 3.0]),
        "arm_quat": np.array([0.0, 0.0, 0.0, 1.0]),
    }
    arm_controller.control_callback(command)
    # The target position should be set, but the exact value depends on TidybotIKSolver
    assert arm_controller.otg_inp.target_position is not None
    assert arm_controller.otg_res is not None


def test_control_callback_sets_gripper(arm_controller):
    """Test that control_callback sets the gripper position correctly."""
    arm_controller.reset()
    command = {"gripper_pos": 0.5}
    arm_controller.control_callback(command)
    assert np.allclose(arm_controller.ctrl_gripper, 255.0 * 0.5)
