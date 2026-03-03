import os
import torch
import numpy as np
from gail_airl_ppo.buffer import SerializedBuffer
from src.env.myolegs_IL import MyoLegsGAIL
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse
import mujoco
import hydra
from hydra import compose, initialize

def generate_expert_buffer(motion_file, output_path, history_len=3, device="cpu"):
    """
    Loads expert motions and creates a SerializedBuffer by running them through the environment.
    """
    # Initialize and compose Hydra config
    with initialize(config_path="../cfg"):
        cfg = compose(config_name="config", overrides=[
            f"env=myolegs_gail",
            f"run.motion_file={motion_file}",
            f"run.num_motions=1000",
            f"run.headless=True",
            f"env.history_len={history_len}"
        ])
    
    # Initialize environment
    env = MyoLegsGAIL(cfg)
    env.sample_motions() # Trigger initial loading of motions
    
    states = []
    
    num_motions = env.motion_lib.num_all_motions()
    print(f"Processing {num_motions} motions...")
    
    for m_id in range(num_motions):
        # Reset environment with specific motion
        obs, _ = env.reset(options={"motion_id": m_id, "start_time": 0})
        
        motion_len = env.motion_lib.get_motion_length(m_id)
        fps = env.motion_lib._motion_fps[m_id]
        num_frames = int(motion_len * fps)
        
        print(f"Motion {m_id}: {num_frames} frames")
        
        # Clear/Init history
        env.history_buffer.clear()
        
        for f in range(num_frames):
            # Get expert state for this frame
            sim_time = f / fps
            ref_dict = env.get_state_from_motionlib_cache(
                np.array([m_id]), np.array([sim_time]), env.global_offset
            )
            
            # Set simulation state to expert state
            env.mj_data.qpos[:] = ref_dict.qpos[0]
            env.mj_data.qvel[:] = ref_dict.qvel[0]
            mujoco.mj_forward(env.mj_model, env.mj_data)
            
            # Get observation (this handles proprioception + task + history padding)
            # We want the concatenated history observation that the discriminator sees.
            current_obs = env.compute_observations()
            env.history_buffer.append(current_obs)
            
            if len(env.history_buffer) == history_len:
                hist_obs = np.concatenate(list(env.history_buffer))
                states.append(hist_obs)
            elif len(env.history_buffer) < history_len:
                # Pad for early frames
                temp_buffer = list(env.history_buffer)
                while len(temp_buffer) < history_len:
                    temp_buffer.insert(0, temp_buffer[0])
                hist_obs = np.concatenate(temp_buffer)
                states.append(hist_obs)

    states = np.array(states, dtype=np.float32)
    # actions = np.zeros((states.shape[0], 0), dtype=np.float32) # Removed: State-only GAIL
    
    print(f"Buffer size: {states.shape}")
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # User updated GAILDiscrim to accept state-only expert data.
    # SerializedBuffer expects 'state' and 'action'.
    torch.save({
        'state': torch.from_numpy(states),
        'action': torch.zeros((states.shape[0], 0))
    }, output_path)
    
    print(f"Expert buffer saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", type=str, default="amputee_data/training_data")
    parser.add_argument("--output_path", type=str, default="data/buffers/expert_buffer.pth")
    parser.add_argument("--history_len", type=int, default=3)
    args = parser.parse_args()
    
    generate_expert_buffer(args.motion_file, args.output_path, args.history_len)
