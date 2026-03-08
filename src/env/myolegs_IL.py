# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.

import os
import joblib
import numpy as np
from collections import OrderedDict, deque
from omegaconf import DictConfig, ListConfig
from typing import Dict, Iterator, Optional, Tuple
import scipy
import torch
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as sRot
from torch.optim import Adam

from pathlib import Path
import sys
path_root = Path(__file__).resolve().parents[2]
sys.path.append(str(path_root))

from src.env.myolegs_gail_task import MyoLegsGailTask
from src.utils.visual_capsule import add_visual_capsule
from src.env.myolegs_gail_env import get_actuator_names
from src.KinesisCore.prostwalk_core import ProstWalkCore
from gail_airl_ppo.network import GAILDiscrim

import logging

logger = logging.getLogger(__name__)

class MyoLegsGAIL(MyoLegsGailTask):
    """
    MyoLegsRL focuses on GAIL training using OpenSim expert data.
    It returns a 32D joint-based observation (16 angles, 16 velocities) 
    with temporal history (s_t, s_t-1, s_t-2).
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.dtype = np.float32
        
        self.initialize_env_params(cfg)
        self.initialize_run_params(cfg)
        
        self.global_offset = np.zeros([1, 3])
        self.history_len = cfg.env.get("history_len", 3)
        self.history_buffer = deque(maxlen=self.history_len)

        super().__init__(cfg)
        
        self.setup_motionlib()
        
        obs_size = self.get_obs_size() 
        self.gail_disc = GAILDiscrim(
            state_shape=(obs_size,),
            action_shape=(0,), # State-only GAIL
            hidden_units=cfg.env.get("gail_hidden_units", (256, 256)),
            state_only=True
        ).to(cfg.run.get("device", "cpu"))
        
        self.optim_disc = Adam(self.gail_disc.parameters(), lr=cfg.env.get("gail_lr", 1e-4))
        self.curr_power_usage = []

    def setup_motionlib(self):
        """Initializes the motion library using ProstWalkCore."""
        joint_names = [self.mj_model.joint(i).name for i in range(self.mj_model.njnt)]
        self.motion_lib = ProstWalkCore(
            self.cfg.run, 
            joint_names=joint_names
        )
        logger.info(f"Motion library initialized with {len(self.motion_lib._curr_motion_ids)} motions.")

    def get_disc_obs(self) -> np.ndarray:
        """
        Computes the 32D raw observation (16 angles + 16 velocities)
        matching the expert data format.
        """
        qpos = self.mj_data.qpos
        qvel = self.mj_data.qvel
        
        # 1. Root pos + rot (Euler xyz)
        quat = qpos[3:7] # w, x, y, z
        r = sRot.from_quat([quat[1], quat[2], quat[3], quat[0]]) # MJ -> scipy
        pelvis_euler = r.as_euler('xyz') 
        
        angles = np.zeros(16, dtype=self.dtype)
        angles[0:3] = qpos[0:3] # tx, ty, tz
        angles[3:6] = pelvis_euler # tilt, list, rot
        
        # Joints: hip_flexion_r=7, hip_adduction_r=8, hip_rotation_r=9,
        # osl_knee_angle_r=14, osl_ankle_angle_r=15,
        # hip_flexion_l=16, hip_adduction_l=17, hip_rotation_l=18,
        # knee_angle_l=21, ankle_angle_l=24
        angles[6:9] = qpos[[7, 8, 9]]
        angles[9:11] = qpos[[14, 15]]
        angles[11:14] = qpos[[16, 17, 18]]
        angles[14:16] = qpos[[21, 24]]
        
        # 2. Velocities
        vels = np.zeros(16, dtype=self.dtype)
        vels[0:6] = qvel[0:6] # Root lin + ang
        vels[6:9] = qvel[[6, 7, 8]]
        vels[9:11] = qvel[[13, 14]]
        vels[11:14] = qvel[[15, 16, 17]]
        vels[14:16] = qvel[[20, 23]]
        
        return np.concatenate([angles, vels])

    def compute_observations(self) -> np.ndarray:
        """Returns the concatenated temporal history of disc observations."""
        raw_obs = self.get_disc_obs()
        self.history_buffer.append(raw_obs)
        
        # Pad if buffer is not full
        hist = list(self.history_buffer)
        while len(hist) < self.history_len:
            hist.insert(0, hist[0])
            
        return np.concatenate(hist)

    def get_obs(self) -> np.ndarray:
        """Returns the current observation."""
        return self.compute_observations()

    def get_obs_size(self) -> int:
        return 32 * self.history_len

    def get_task_obs_size(self) -> int:
        return 0

    def reset_task(self, options=None):
        """Resets task-specific state."""
        self.history_buffer.clear()
        
        if options is not None and "motion_id" in options:
            self._sampled_motion_ids[:] = options["motion_id"]
        else:
            self._sampled_motion_ids[:] = self.motion_lib.sample_motions()
            
        if options is not None and "start_time" in options:
            self._motion_start_times[:] = options["start_time"]
        else:
            self._motion_start_times[:] = 0

    def compute_task_obs(self) -> np.ndarray:
        return np.empty(0, dtype=self.dtype)

    def draw_task(self):
        pass

    def create_task_visualization(self):
        pass

    def compute_reward(self, action: Optional[np.ndarray] = None) -> float:
        """GAIL Reward using the Discriminator."""
        obs = self.get_obs()
        device = next(self.gail_disc.parameters()).device
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        a_tensor = torch.zeros((1, 0), device=device)
        
        with torch.no_grad():
            im_reward = self.gail_disc.calculate_reward(obs_tensor, a_tensor).item()
            
        reward = im_reward + 0.1
        self.reward_info = {"imitation_reward_gail": im_reward, "total_reward": reward}
        return reward

    def compute_energy_reward(self, ctrl: np.ndarray) -> float:
        """Computes a simple energy-based penalty based on control effort."""
        return -np.sum(np.square(ctrl))

    def compute_reset(self) -> Tuple[bool, bool]:
        """Basic stability and time-based reset."""
        fell = self.mj_data.qpos[2] < 0.5
        truncated = self.cur_t >= self.max_episode_length
        return fell, truncated

    def initialize_env_params(self, cfg: DictConfig) -> None:
        self.max_episode_length = cfg.env.get("max_episode_length", 300)
        self.muscle_condition = cfg.env.get("muscle_condition", "")

    def initialize_run_params(self, cfg: DictConfig) -> None:
        self.motion_start_idx = cfg.run.motion_id
        self.num_motion_max = cfg.run.num_motions
        self.motion_file = cfg.run.motion_file
        self.initial_pose_file = cfg.run.initial_pose_file
        self.device = cfg.run.get("device", "cpu")
        self.num_threads = cfg.run.get("num_threads", 1)
        self._sampled_motion_ids = np.zeros(self.num_threads, dtype=np.int32)
        self._motion_start_times = np.zeros(self.num_threads, dtype=np.float32)