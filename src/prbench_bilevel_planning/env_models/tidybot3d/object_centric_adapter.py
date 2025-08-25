"""Object-centric adapter for TidyBot3D environment.

This module provides an adapter that converts TidyBot3D observations to
object-centric format compatible with bilevel planning models.
"""

import numpy as np
from gymnasium.spaces import Box
from relational_structs import Object, ObjectCentricState, Type
from relational_structs.spaces import ObjectCentricBoxSpace, ObjectCentricStateSpace


def create_object_centric_observation_space() -> ObjectCentricBoxSpace:
    """Create an object-centric observation space for TidyBot3D."""
    # Define types
    robot_type = Type("robot")
    cube_type = Type("cube")
    target_type = Type("target")
    
    # Create objects for the space
    robot = Object("robot", robot_type)
    cubes = [Object(f"cube{i+1}", cube_type) for i in range(3)]
    targets = [Object(f"target{i+1}", target_type) for i in range(3)]
    objects = [robot] + cubes + targets
    
    # Define features for each type
    type_features = {
        robot_type: ["x", "y", "theta", "arm_x", "arm_y", "arm_z", "gripper"],
        cube_type: ["x", "y", "z"],
        target_type: ["x", "y", "z"],
    }
    
    # Create the object-centric space
    return ObjectCentricBoxSpace(objects, type_features)


def observation_to_object_centric_state(obs: np.ndarray) -> ObjectCentricState:
    """Convert TidyBot3D observation to object-centric state.
    
    The TidyBot3D observation is a 32-dimensional vector with structure:
    - obs[0:3]: base_pose (x, y, theta)
    - obs[3:6]: arm_pos (x, y, z) 
    - obs[6:10]: arm_quat (x, y, z, w)
    - obs[10:11]: gripper_pos
    - obs[11:14]: cube1_pos (x, y, z)
    - obs[14:18]: cube1_quat (x, y, z, w)
    - obs[18:21]: cube2_pos (x, y, z)
    - obs[21:25]: cube2_quat (x, y, z, w)
    - obs[25:28]: cube3_pos (x, y, z)
    - obs[28:32]: cube3_quat (x, y, z, w)
    
    Note: In TidyBot3D ground scene, there are no explicit "target" objects.
    The targets are typically goal positions for placing cubes, not physical objects.
    For bilevel planning, we'll create virtual target objects at fixed locations.
    """
    # Create types
    robot_type = Type("robot")
    cube_type = Type("cube")
    target_type = Type("target")
    
    # Create objects
    robot = Object("robot", robot_type)
    cube1 = Object("cube1", cube_type)
    cube2 = Object("cube2", cube_type)
    cube3 = Object("cube3", cube_type)
    target1 = Object("target1", target_type)
    target2 = Object("target2", target_type)
    target3 = Object("target3", target_type)
    
    # Parse observation vector (32 dimensions)
    # Robot state: combine base and arm info
    robot_state = np.array([
        obs[0],     # base_x
        obs[1],     # base_y  
        obs[2],     # base_theta
        obs[3],     # arm_x
        obs[4],     # arm_y
        obs[5],     # arm_z
        obs[10],    # gripper_pos
    ])
    
    # Extract cube positions (ignore quaternions for now as we only use positions)
    cube1_pos = obs[11:14]  # cube1 position
    cube2_pos = obs[18:21]  # cube2 position  
    cube3_pos = obs[25:28]  # cube3 position
    
    # Create virtual target positions (typical goal locations for ground scene)
    # These are reasonable target locations on the ground plane
    target1_pos = np.array([1.5, 0.5, 0.02])   # Target for cube1
    target2_pos = np.array([1.5, 0.0, 0.02])   # Target for cube2
    target3_pos = np.array([1.5, -0.5, 0.02])  # Target for cube3
    
    # Create data dictionary with the features we can extract
    data = {
        robot: robot_state,    # Robot state: 7 features
        cube1: cube1_pos,      # First cube position: 3 features
        cube2: cube2_pos,      # Second cube position: 3 features
        cube3: cube3_pos,      # Third cube position: 3 features
        target1: target1_pos,  # First target position: 3 features
        target2: target2_pos,  # Second target position: 3 features
        target3: target3_pos,  # Third target position: 3 features
    }
    
    # Create type features
    type_features = {
        robot_type: ["x", "y", "theta", "arm_x", "arm_y", "arm_z", "gripper"],
        cube_type: ["x", "y", "z"],
        target_type: ["x", "y", "z"],
    }
    
    return ObjectCentricState(data, type_features)


def create_tidybot3d_bilevel_planning_models(
    observation_space: Box, action_space: Box, **kwargs
):
    """Create bilevel planning models for TidyBot3D using the adapter."""
    # Import here to avoid circular imports
    from .ground import create_bilevel_planning_models
    
    # Create object-centric observation space
    object_centric_obs_space = create_object_centric_observation_space()
    
    # Create the models using the object-centric space
    return create_bilevel_planning_models(
        object_centric_obs_space, action_space, **kwargs
    ) 