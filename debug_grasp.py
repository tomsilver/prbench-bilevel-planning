#!/usr/bin/env python3
"""Debug script to understand grasping behavior."""

import numpy as np
import prbench
from prbench_bilevel_planning.env_models import create_bilevel_planning_models

prbench.register_all_environments()

def debug_grasp():
    env = prbench.make("prbench/StickButton2D-b1-v0")
    env_models = create_bilevel_planning_models(
        "stickbutton2d", env.observation_space, env.action_space, num_buttons=1
    )
    
    obs0, _ = env.reset(seed=123)
    state0 = env_models.observation_to_state(obs0)
    abstract_state = env_models.state_abstractor(state0)
    obj_name_to_obj = {o.name: o for o in abstract_state.objects}
    robot = obj_name_to_obj["robot"]
    stick = obj_name_to_obj["stick"]
    
    print("=== Initial State ===")
    print(f"Robot position: ({state0.get(robot, 'x'):.3f}, {state0.get(robot, 'y'):.3f})")
    print(f"Robot vacuum: {state0.get(robot, 'vacuum'):.3f}")
    print(f"Robot arm_joint: {state0.get(robot, 'arm_joint'):.3f}")
    print(f"Stick position: ({state0.get(stick, 'x'):.3f}, {state0.get(stick, 'y'):.3f})")
    print(f"Stick dimensions: {state0.get(stick, 'width'):.3f} x {state0.get(stick, 'height'):.3f}")
    print(f"Abstract state atoms: {[str(atom) for atom in abstract_state.atoms]}")
    
    # Get the skill
    skill_name_to_skill = {s.operator.name: s for s in env_models.skills}
    PickStickFromNothing = skill_name_to_skill["PickStickFromNothing"]
    pick_skill = PickStickFromNothing.ground((robot, stick))
    
    # Sample parameters and reset controller
    rng = np.random.default_rng(123)
    controller = pick_skill.controller
    params = controller.sample_parameters(state0, rng)
    print(f"\nSampled params: {params}")
    controller.reset(state0, params)
    
    # Execute the skill
    state = state0
    for step in range(200):
        action = controller.step()
        print(f"Step {step}: action = {action}")
        executable = env_models.action_to_executable(action)
        obs, _, _, _, _ = env.step(executable)
        next_state = env_models.observation_to_state(obs)
        controller.observe(next_state)
        state = next_state
        
        # Print key state info every few steps
        if step % 20 == 0 or controller.terminated():
            print(f"  Robot pos: ({state.get(robot, 'x'):.3f}, {state.get(robot, 'y'):.3f})")
            print(f"  Robot vacuum: {state.get(robot, 'vacuum'):.3f}")
            print(f"  Robot arm_joint: {state.get(robot, 'arm_joint'):.3f}")
            abstract_state = env_models.state_abstractor(state)
            grasped_atoms = [atom for atom in abstract_state.atoms if atom.predicate.name == "Grasped"]
            handempty_atoms = [atom for atom in abstract_state.atoms if atom.predicate.name == "HandEmpty"]
            print(f"  Grasped: {grasped_atoms}")
            print(f"  HandEmpty: {handempty_atoms}")
        
        if controller.terminated():
            print(f"Controller terminated at step {step}")
            break
    
    print("\n=== Final State ===")
    final_abstract_state = env_models.state_abstractor(state)
    print(f"Final abstract state atoms: {[str(atom) for atom in final_abstract_state.atoms]}")
    
    env.close()

if __name__ == "__main__":
    debug_grasp()