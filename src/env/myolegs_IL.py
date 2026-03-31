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
        self.history_len = cfg.run.get("history_len", 6)
        self.history_buffer = deque(maxlen=self.history_len)

        super().__init__(cfg)
        
        self.setup_motionlib()
        
        # Discriminator receives same observations as actor (33D per frame)
        obs_size = self.get_obs_size() 
        self.gail_disc = GAILDiscrim(
            state_shape=(obs_size,),
            action_shape=(0,), # State-only GAIL
            hidden_units=cfg.env.get("gail_hidden_units", (256, 256)),
            state_only=True
        ).to(cfg.run.get("device", "cpu"))
        
        self.optim_disc = Adam(self.gail_disc.parameters(), lr=cfg.env.get("gail_lr", 1e-4))

    def setup_motionlib(self):
        """Initializes the motion library using ProstWalkCore."""
        joint_names = [self.mj_model.joint(i).name for i in range(self.mj_model.njnt)]
        self.motion_lib = ProstWalkCore(
            self.cfg.run, 
            joint_names=joint_names
        )
        self.motion_lib.load_motions(self.cfg.run)
        logger.info(f"Motion library initialized with {len(self.motion_lib.curr_motion_keys)} motions.")

    def get_disc_obs(self) -> np.ndarray:
        """
        Computes the 30D raw observation (13 angles + 16 velocities + 1 target speed)
        matching the expert data format, excluding pelvis translation.
        """
        qpos = self.mj_data.qpos
        qvel = self.mj_data.qvel
        
        # 1. Root rotation (Euler xyz) - NO translation
        quat = qpos[3:7] # w, x, y, z
        r = sRot.from_quat([quat[1], quat[2], quat[3], quat[0]]) # MJ -> scipy
        pelvis_euler = r.as_euler('xyz') 
        
        angles = np.zeros(13, dtype=self.dtype)
        angles[0:3] = pelvis_euler # tilt, list, rot
        
        # Joints: hip_flexion_r=7, hip_adduction_r=8, hip_rotation_r=9,
        # osl_knee_angle_r=14, osl_ankle_angle_r=15,
        # hip_flexion_l=16, hip_adduction_l=17, hip_rotation_l=18,
        # knee_angle_l=21, ankle_angle_l=24
        angles[3:6] = qpos[[7, 8, 9]]
        angles[6:8] = qpos[[14, 15]]
        angles[8:11] = qpos[[16, 17, 18]]
        angles[11:13] = qpos[[21, 24]]
        
        # 2. Velocities (ALL kept, including root linear velocity)
        vels = np.zeros(16, dtype=self.dtype)
        vels[0:6] = qvel[0:6] # Root lin + ang
        vels[6:9] = qvel[[6, 7, 8]]
        vels[9:11] = qvel[[13, 14]]
        vels[11:14] = qvel[[15, 16, 17]]
        vels[14:16] = qvel[[20, 23]]

        self.curr_proprioception = angles
        
        # 3. Target Speed
        target_speed = np.array([self.target_speed], dtype=self.dtype)
        
        return np.concatenate([angles, vels, target_speed])

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
        """Observation size: 30D per frame (13 angles + 16 velocities + 1 speed) * history."""
        return 30 * self.history_len

    def get_task_obs_size(self) -> int:
        return 0

    def init_myolegs(self):
        """
        Initializes the MyoLegs environment from a pre-configured valid pose.
        """
        self.mj_data.qpos[:] = 0
        self.mj_data.qvel[:] = 0
        
        # 'stand' keyframe from myolegs_OSL_KA.xml
        self.mj_data.qpos[2] = 0.95
        self.mj_data.qpos[3:7] = np.array([0.707388, 0, 0, -0.706825])
            
        mujoco.mj_kinematics(self.mj_model, self.mj_data)

    def reset_task(self, options=None):
        """Resets task-specific state."""
        self.history_buffer.clear()
        
        # Pick a target speed from the expert library
        if options is not None and "target_speed" in options:
            self.target_speed = options["target_speed"]
        elif self.cfg.run.test and getattr(self.cfg.run, "eval_target_speed", None) is not None:
            self.target_speed = self.cfg.run.eval_target_speed
        else:
            self.target_speed = np.random.choice(self.motion_lib.available_speeds)
            
        self.biomechanics_data = []
        logger.info(f"Target speed for this episode: {self.target_speed}")

    def compute_task_obs(self) -> np.ndarray:
        return np.empty(0, dtype=self.dtype)

    def draw_task(self):
        pass

    def create_task_visualization(self):
        pass

    def compute_reward(self, action: Optional[np.ndarray] = None) -> float:
        """GAIL Reward using the Discriminator + Velocity Matching."""
        obs = self.get_obs()
        device = next(self.gail_disc.parameters()).device
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        a_tensor = torch.zeros((1, 0), device=device)
        
        with torch.no_grad():
            im_reward = self.gail_disc.calculate_reward(obs_tensor, a_tensor).item()
            
        vel_reward = self.compute_velocity_reward()
        upright_reward = self.compute_upright_reward()
        
        # Split Energy Reward
        muscle_effort = self.compute_muscle_effort(action)
        motor_effort = self.compute_motor_effort(action)
        
        w_muscle = self.cfg.run.get("muscle_effort_weight", 0.01)
        w_motor = self.cfg.run.get("motor_effort_weight", 0.01)
        
        reward = 1 * im_reward + 0.2 * vel_reward + 0.3 * upright_reward - 0.01 * muscle_effort - 0.03 * motor_effort
        
        self.reward_info = {
            "imitation_reward_gail": im_reward, 
            "velocity_reward": vel_reward,
            "upright_reward": upright_reward,
            "muscle_effort": muscle_effort,
            "motor_effort": motor_effort,
            "total_reward": reward
        }
        return reward

    def compute_velocity_reward(self) -> float:
        """Rewards matching the 3D velocity vector to the target direction vector."""
        # Root lin vel is in qvel[0:3]
        actual_v = self.mj_data.qvel[0:3]
        target_v = np.array([self.target_speed, 0, 0])
        
        # Dot product reward
        # v_actual · v_target
        dot = np.dot(actual_v, target_v)
        # Normalize by target speed squared to keep range around 0-1
        vel_reward = np.exp(-2 * (self.target_speed - dot/self.target_speed)**2) if self.target_speed > 0.01 else 1.0
        return vel_reward

    def compute_muscle_effort(self, action: np.ndarray) -> float:
        """Computes effort penalty for biological muscles."""
        if action is None: return 0.0
        # self.muscle_idx are the indices in the full action vector
        muscle_acts = action[self.muscle_idx]
        return np.sum(np.square(muscle_acts))

    def compute_motor_effort(self, action: np.ndarray) -> float:
        """Computes effort penalty for prosthetic motors."""
        if action is None: return 0.0
        motor_acts = action[self.motor_idx]
        return np.sum(np.square(motor_acts))

    def compute_upright_reward(self) -> float:
        """
        Computes the reward for maintaining an upright posture.

        The reward is based on the angles of tilt in the forward and sideways directions, 
        calculated using trigonometric components of the root tilt.

        Returns:
            float: The upright reward, where a value close to 1 indicates a nearly upright posture.
        """
        root_rot_euler = self.curr_proprioception[0:3]
        upright_trigs = np.array([np.cos(root_rot_euler[0]), np.sin(root_rot_euler[0]), np.cos(root_rot_euler[1]), np.sin(root_rot_euler[1])])
        fall_forward = np.angle(upright_trigs[0] + 1j * upright_trigs[1])
        fall_sideways = np.angle(upright_trigs[2] + 1j * upright_trigs[3])
        upright_reward = np.exp(-3 * (fall_forward ** 2 + fall_sideways ** 2))
        return upright_reward

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

    def record_biomechanics(self):
        """Records biomechanics state at the current timestep testing."""
        if not self.cfg.run.test or not getattr(self.cfg.run, "record_biomechanics", False):
            return
            
        # Record heights for ground clearance diagnostics
        # osl_foot_assembly (12), calcn_l (16)
        # We use xpos[index, 2] for the Z-coordinate
        right_foot_z = self.mj_data.xpos[12, 2]
        left_foot_z = self.mj_data.xpos[16, 2]
        
        # Get ankle heights as well (joints 9 and 18)
        # Note: bodies associated with these joints might be better. 
        # For osl_ankle, it's body 'osl_ankle_assembly' (11).
        # For ankle_l, it's body 'talus_l' (15).
        right_ankle_z = self.mj_data.xpos[11, 2]
        left_ankle_z = self.mj_data.xpos[15, 2]

        # Calculate Net Torque (aggregate joint moments) as requested:
        # qfrc_actuator + qfrc_applied + qfrc_passive - qfrc_bias
        # We use getattr for robustness across MuJoCo versions
        q_act = self.mj_data.qfrc_actuator.copy()
        q_app = getattr(self.mj_data, "qfrc_applied", np.zeros_like(q_act)).copy()
        q_pas = getattr(self.mj_data, "qfrc_passive", np.zeros_like(q_act)).copy()
        q_bia = getattr(self.mj_data, "qfrc_bias", np.zeros_like(q_act)).copy()
        
        net_torque = q_act + q_app + q_pas - q_bia

        data = {
            "qpos": self.mj_data.qpos.copy(),
            "qvel": self.mj_data.qvel.copy(),
            "ctrl": self.mj_data.ctrl.copy(),
            "qfrc_actuator": net_torque, # Rename to qfrc_actuator for plotter compatibility or use new key
            "qfrc_actuator_only": self.mj_data.qfrc_actuator.copy(),
            "actuator_force": self.mj_data.actuator_force.copy(),
            "actuator_activation": getattr(self.mj_data, "actuator_activation", getattr(self.mj_data, "act", np.zeros(0))).copy(),
            "impedance": getattr(self, "last_impedance", {}).copy(),
            "heights": {
                "right_foot": right_foot_z,
                "left_foot": left_foot_z,
                "right_ankle": right_ankle_z,
                "left_ankle": left_ankle_z
            }
        }
        self.biomechanics_data.append(data)

    def post_physics_step(self, action):
        """Overrides base post_physics_step to include biomechanics tracking."""
        obs, reward, terminated, truncated, info = super().post_physics_step(action)
        self.record_biomechanics()
        
        if terminated or truncated:
            if getattr(self.cfg.run, "record_biomechanics", False):
                info["biomechanics_data"] = self.biomechanics_data
                
        return obs, reward, terminated, truncated, info