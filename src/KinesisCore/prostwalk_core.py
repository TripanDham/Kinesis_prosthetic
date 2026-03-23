# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.
#
# Licensed under the BSD 3-Clause License.
#
# This file contains code adapted from:
#
# 1. SMPLSim (https://github.com/ZhengyiLuo/SMPLSim)
#    Copyright (c) 2024 Zhengyi Luo
#    Licensed under the BSD 3-Clause License.

import os
import sys
from typing import List, Tuple, Union

sys.path.append(os.getcwd())

import numpy as np
from tqdm import tqdm

import torch
import torch.multiprocessing as mp

import joblib
import re
import pandas as pd
from pathlib import Path

from scipy.spatial.transform import Rotation as sRot
import random
random.seed(0)
from src.utils.torch_utils import to_torch
from easydict import EasyDict

# from src.KinesisCore.forward_kinematics import ForwardKinematics

torch.set_num_threads(1)

class ProstWalkCore:

    def __init__(self, config, joint_names=None):
        self.config = config
        self.dtype = np.float32
        self.joint_names = joint_names

        self.load_data(config.motion_file)
        self._curr_motion_ids = np.arange(self._num_unique_motions)
        self._sampling_batch_prob = (
            np.ones(self._num_unique_motions) / self._num_unique_motions
        )
        self._velocity_groups = self._bunch_by_velocity()
        # SMPL features are disabled
        self.fk_model = None

    def load_data(self, filepath: str) -> None:
        """
        Loads motion data from a given pickle file or a directory of OpenSim .mot files.
        """
        if os.path.isdir(filepath):
            self.motion_data = self._load_opensim_dir(filepath)
        else:
            self.motion_data = joblib.load(filepath)
        self._num_unique_motions = len(self.motion_data.keys())
        self._curr_motion_ids = np.array(list(range(self._num_unique_motions)))

    def _load_opensim_dir(self, directory: str) -> dict:
        """Loads all .mot files in a directory and returns a dictionary of motion data."""
        cache_file = os.path.join(directory, "processed_motions.joblib")
        if os.path.exists(cache_file):
            print(f"Loading cached OpenSim data from {cache_file}")
            return joblib.load(cache_file)
            
        motion_data = {}
        mot_files = list(Path(directory).glob("*.mot"))
        print(f"Parsing {len(mot_files)} .mot files...")
        for mot_file in tqdm(mot_files):
            name = mot_file.stem
            data = self._parse_mot(str(mot_file), self.joint_names)
            motion_data[name] = data
        
        # Cache for next time
        print(f"Caching processed data to {cache_file}")
        joblib.dump(motion_data, cache_file)
        return motion_data

    def _parse_mot(self, filepath: str, mu_joint_names: List[str] = None) -> dict:
        """Parses an OpenSim .mot file into a dictionary compatible with ProstWalkCore."""
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        header_end = 0
        in_degrees = True
        for i, line in enumerate(lines):
            if 'inDegrees' in line:
                in_degrees = 'yes' in line.lower()
            if 'endheader' in line:
                header_end = i + 1
                break
        
        df = pd.read_csv(filepath, sep='\s+', skiprows=header_end)
        
        # Minimally map Pelvis and Time
        fps = 1.0 / (df['time'].iloc[1] - df['time'].iloc[0]) if len(df) > 1 else 30
        
        # Detect units
        unit_scale = np.pi / 180.0 if in_degrees else 1.0
        
        # ============================================================
        # Frame rotation: OpenSim Y-up -> MuJoCo Z-up
        # The original rotation applied (to go from MuJoCo to OpenSim) was:
        #   1st: Rx(-π/2)  2nd: Ry(π/2)
        # So the INVERSE (OpenSim -> MuJoCo) is:
        #   1st: Ry(-π/2)  2nd: Rx(π/2)
        # ============================================================
        # R_y_neg = sRot.from_euler('y', -np.pi/2)
        R_x_pos = sRot.from_euler('x', np.pi/2)
        R_frame = R_x_pos
        R_frame_matrix = R_frame.as_matrix()  # 3x3
        
        # --- Rotate Pelvis Translation ---
        pelvis_trans_raw = df[['pelvis_tx', 'pelvis_ty', 'pelvis_tz']].values
        pelvis_trans = (R_frame_matrix @ pelvis_trans_raw.T).T  # (N, 3)
        
        # --- Rotate Pelvis Orientation ---
        # Convert euler angles to quaternions in OpenSim frame, then rotate into MuJoCo frame
        pelvis_euler = df[['pelvis_tilt', 'pelvis_list', 'pelvis_rotation']].values * unit_scale
        pelvis_rot_opensim = sRot.from_euler('xyz', pelvis_euler)
        pelvis_rot_mujoco = R_frame * pelvis_rot_opensim  # Pre-multiply by frame rotation
        pelvis_quat = pelvis_rot_mujoco.as_quat()  # [x, y, z, w]
        
        if mu_joint_names is not None:
             # Identify actual hinges/slides vs the root freejoint
             # In MuJoCo, root is typically index 0 and has 'root' in the name
             real_joint_names = [n for n in mu_joint_names if 'root' not in n.lower()]
             joint_angles = np.zeros((len(df), len(real_joint_names)), dtype=np.float32)
             
             for i, mu_name in enumerate(real_joint_names):
                  if mu_name in df.columns:
                       joint_angles[:, i] = df[mu_name].values * unit_scale
                  else:
                       # Handle cases like socket joints that might be in model but not in .mot
                       pass 
        else:
            # Fallback to old behavior: extract everything except pelvis
            exclude = ['time', 'pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz']
            joint_cols = [c for c in df.columns if c not in exclude]
            joint_angles = df[joint_cols].values * unit_scale
        
        # Combine into qpos-like structure for ProstWalkCore
        # Note: pelvis_quat is typically [x, y, z, w]. 
        # MuJoCo uses [w, x, y, z].
        pelvis_quat_mj = np.roll(pelvis_quat, 1, axis=1) # [x, y, z, w] -> [w, x, y, z]
        
        qpos = np.concatenate([pelvis_trans, pelvis_quat_mj, joint_angles], axis=1)
        
        # Compute qvel (from ROTATED translation and orientation)
        dt = 1.0 / fps
        # 1. Pelvis linear velocity (world frame, already in MuJoCo frame)
        lin_vel = np.diff(pelvis_trans, axis=0) / dt
        
        # Extract nominal speed from filename to add to forward velocity
        # Pattern: _0p6_ -> 0.6
        match = re.search(r'_(\d+p\d+)', os.path.basename(filepath))
        if match:
            motion_speed = float(match.group(1).replace('p', '.'))
            lin_vel[:, 0] += motion_speed
        
        # 2. Pelvis angular velocity (world frame)
        # pelvis_quat is [x, y, z, w] from sRot (already rotated)
        r = sRot.from_quat(pelvis_quat)
        r1 = r[:-1]
        r2 = r[1:]
        # Local angular velocity: r1.inv() * r2
        rel_rot = r1.inv() * r2
        ang_vel_local = rel_rot.as_rotvec() / dt
        # Convert to world frame: omega_world = r1.apply(omega_local)
        ang_vel = r1.apply(ang_vel_local)
        
        # 3. Joint velocities
        joint_vel = np.diff(joint_angles, axis=0) / dt
        
        # Combine into qvel-like structure (lin(3), ang(3), joints(N))
        qvel = np.concatenate([lin_vel, ang_vel, joint_vel], axis=1)
        
        # Match length by padding the last frame
        qvel = np.concatenate([qvel, qvel[-1:]], axis=0)
        
        return {
            'qpos': qpos.astype(np.float32),
            'qvel': qvel.astype(np.float32),
            'fps': fps,
            'pose_aa': np.zeros((qpos.shape[0], 24, 3)), # Placeholder
            'trans_orig': pelvis_trans_raw.astype(np.float32),
        }


    def _bunch_by_velocity(self) -> dict:
        """Bunches motion IDs by velocity extracted from keys (filenames)."""
        groups = {}
        for idx, key in enumerate(self.motion_data.keys()):
            # Search for pattern like _0p6 or _1p2
            match = re.search(r'_(\d+p\d+)', key)
            if match:
                vel_str = match.group(1).replace('p', '.')
                vel = float(vel_str)
            else:
                vel = 0.0 # Default/Unknown
            
            if vel not in groups:
                groups[vel] = []
            groups[vel].append(idx)
        return groups

    @property
    def available_speeds(self):
        return sorted(list(self._velocity_groups.keys()))

    def sample_motions_by_velocity(self, velocity: float, n: int = 1) -> np.ndarray:
        """Samples motions specifically from a velocity group."""
        if velocity not in self._velocity_groups:
            print(f"Warning: Velocity {velocity} not found. Sampling from all.")
            return self.sample_motions(n)
        
        group_indices = self._velocity_groups[velocity]
        return np.random.choice(group_indices, size=n, replace=True)

    def load_motions(
            self,
            m_cfg: dict,
            num_motions: int = None,
            shape_params: List[np.ndarray] = None,
            random_sample: bool = True,
            start_idx: int = 0,
            silent: bool = False,
            specific_idxes: np.ndarray = None,
    ):
        
        motions = []
        motion_lengths = []
        motion_fps_acc = []
        motion_dt = []
        motion_num_frames = []
        motion_bodies = []
        motion_aa = []

        self.num_joints = 24

        if num_motions is not None:
             num_motion_to_load = num_motions
        elif shape_params is not None:
             num_motion_to_load = len(shape_params)
        else:
             num_motion_to_load = self._num_unique_motions
        if specific_idxes is not None:
            if len(specific_idxes) < num_motion_to_load:
                num_motion_to_load = len(specific_idxes)
            if random_sample:
                sample_idxes = np.random.choice(
                    specific_idxes,
                    size=num_motion_to_load,
                    replace=False,
                )
            else:
                sample_idxes = specific_idxes
        else:
            if random_sample:
                sample_idxes = np.random.choice(
                    np.arange(self._num_unique_motions),
                    size=num_motion_to_load,
                    replace=False,
                )
            else:
                sample_idxes = np.remainder(
                    np.arange(start_idx, start_idx + num_motion_to_load),
                    self._num_unique_motions,
                )

        self._curr_motion_ids = sample_idxes
        self.curr_motion_keys = [list(self.motion_data.keys())[i] for i in sample_idxes]
        
        self._sampling_batch_prob = np.ones(len(self._curr_motion_ids)) / len(
            self._curr_motion_ids
        )
        
        motion_data_list = [self.motion_data[self.curr_motion_keys[i]] for i in range(num_motion_to_load)]

        if sys.platform == "darwin":
            num_jobs = 1
        else:
            mp.set_sharing_strategy("file_descriptor")

        manager = mp.Manager()
        queue = manager.Queue()
        num_jobs = min(min(mp.cpu_count(), 64), num_motion_to_load)

        if len(motion_data_list) <= 32 or not self.config.multi_thread or num_jobs <= 8:
            num_jobs = 1

        res_acc = {}

        chunk = np.ceil(len(motion_data_list) / num_jobs).astype(int)
        ids = np.arange(len(motion_data_list))

        jobs = [
            (
                ids[i: i + chunk],
                motion_data_list[i: i + chunk],
                self.config,
            )
            for i in range(0, len(motion_data_list), chunk)
        ]
        for i in range(1, len(jobs)):
            worker_args = (*jobs[i], queue, i)
            worker = mp.Process(target=self.load_motions_worker, args=worker_args)
            worker.start()
        res_acc.update(self.load_motions_worker(*jobs[0], None, 0))
        pbar = tqdm(range(len(jobs) - 1))
        for i in pbar:
            res = queue.get()
            res_acc.update(res)
        pbar = tqdm(range(len(res_acc)))

        for f in pbar:
            curr_motion = res_acc[f]
            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps
            num_frames = curr_motion.global_translation.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)
            motion_aa.append(curr_motion.pose_aa)
            motion_fps_acc.append(motion_fps)
            motion_dt.append(curr_dt)
            motion_num_frames.append(num_frames)
            motions.append(curr_motion)
            motion_lengths.append(curr_len)

            del curr_motion

        self._motion_lengths = np.array(motion_lengths).astype(self.dtype)
        self._motion_fps = np.array(motion_fps_acc).astype(self.dtype)
        self._motion_aa = np.concatenate(motion_aa, axis=0).astype(self.dtype)
        self._motion_dt = np.array(motion_dt).astype(self.dtype)
        self._motion_num_frames = np.array(motion_num_frames)
        self._num_motions = len(motions)

        self.gts = np.concatenate(
            [m.global_translation for m in motions], axis=0
        ).astype(self.dtype)
        self.grs = np.concatenate(
            [m.global_rotation for m in motions], axis=0
        ).astype(self.dtype)
        self.lrs = np.concatenate(
            [m.local_rotation for m in motions], axis=0
        ).astype(self.dtype)
        self.grvs = np.concatenate(
            [m.global_root_velocity for m in motions], axis=0
        ).astype(self.dtype)
        self.gravs = np.concatenate(
            [m.global_root_angular_velocity for m in motions], axis=0
        ).astype(self.dtype)
        self.gavs = np.concatenate(
            [m.global_angular_velocity for m in motions], axis=0
        ).astype(self.dtype)
        self.gvs = np.concatenate([m.global_velocity for m in motions], axis=0).astype(
            self.dtype
        )
        self.dvs = np.concatenate([m.dof_vels for m in motions], axis=0).astype(
            self.dtype
        )
        self.dof_pos = np.concatenate([m.dof_pos for m in motions], axis=0).astype(
            self.dtype
        )
        self.qpos = np.concatenate([m.qpos for m in motions], axis=0).astype(self.dtype)
        self.qvel = np.concatenate([m.qvel for m in motions], axis=0).astype(self.dtype)

        lengths = self._motion_num_frames
        lengths_shifted = np.roll(lengths, 1, axis=0)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)
        self.motion_ids = np.arange(len(motions))
        self.num_bodies = self.num_joints

        num_motions = self._num_motions
        total_len = sum(self._motion_lengths)
        print(
            f"###### Sampling {num_motions:d} motions:",
            sample_idxes[:5],
            self.curr_motion_keys[:5],
            f"total length of {total_len:.3f}s and {self.gts.shape[0]} frames.",
        )

        return motions

    def load_motions_worker(
            self,
            ids: np.ndarray,
            motion_data_list: List[dict],
            config: dict,
            queue: Union[mp.Queue, None],
            pid: int,
    ):
        np.random.seed(np.random.randint(5000) * pid)
        res = {}
        for f in range(len(motion_data_list)):
            curr_id = ids[f]
            motion_data = motion_data_list[f]
            
            # Handle OpenSim data that already has qpos/qvel
            if 'qpos' in motion_data and 'qvel' in motion_data:
                fk_motion = EasyDict({
                    'qpos': motion_data['qpos'],
                    'qvel': motion_data['qvel'],
                    'fps': motion_data.get('fps', 30),
                    'global_translation': motion_data.get('global_translation', np.zeros((motion_data['qpos'].shape[0], 24, 3))),
                    'global_rotation': motion_data.get('global_rotation', np.zeros((motion_data['qpos'].shape[0], 24, 4))),
                    'local_rotation': motion_data.get('local_rotation', np.zeros((motion_data['qpos'].shape[0], 24, 4))),
                    'global_root_velocity': motion_data['qvel'][:, :3],
                    'global_root_angular_velocity': motion_data['qvel'][:, 3:6],
                    'global_angular_velocity': motion_data.get('global_angular_velocity', np.zeros((motion_data['qpos'].shape[0], 24, 3))),
                    'global_velocity': motion_data.get('global_velocity', np.zeros((motion_data['qpos'].shape[0], 24, 3))),
                    'dof_pos': motion_data['qpos'][:, 7:],
                    'dof_vels': motion_data['qvel'][:, 6:],
                    'pose_aa': motion_data.get('pose_aa', np.zeros((motion_data['qpos'].shape[0], 24, 3))),
                })
                # Minimally fill global_translation with root pos if missing
                if np.all(fk_motion.global_translation == 0):
                    fk_motion.global_translation[:, 0, :] = fk_motion.qpos[:, :3]
                
                res[curr_id] = fk_motion
                continue

            fps = motion_data.get("fps", 30)
            motion_length = motion_data["pose_aa"].shape[0]

            trans = (
                to_torch(
                    motion_data["trans"]
                    if "trans" in motion_data
                    else motion_data["trans_orig"]
                ).float().clone()
            )

            pose_aa = to_torch(motion_data["pose_aa"]).float().clone()
            if pose_aa.shape[1] == 156:
                pose_aa = torch.cat(
                    [pose_aa[:, :66], torch.zeros((pose_aa.shape[0], 6))], dim=1
                ).reshape(-1, 24, 3)
            elif pose_aa.shape[1] == 72:
                pose_aa = pose_aa.reshape(-1, 24, 3)

            B, J, N = pose_aa.shape

            if config.randomize_heading:
                random_rot = np.zeros(3)
                random_rot[2] = np.pi * (2 * np.random.random() - 1.0)
                random_heading_rot = sRot.from_euler("xyz", random_rot)
                pose_aa[:, 0, :] = torch.tensor(
                    (
                        random_heading_rot * sRot.from_rotvec(pose_aa[:, 0, :])
                    ).as_rotvec()
                )
                trans = torch.matmul(
                    trans.float(),
                    torch.from_numpy(random_heading_rot.as_matrix().T).float(),
                )

            trans, trans_fix = self.fix_trans_height(
                pose_aa,
                trans
            )

            self.fk_model.update_model(betas=torch.zeros((1,10)), dt = 1/fps)

            fk_motion = self.fk_model.fk_batch(
                pose_aa[None,],
                trans[None,],
            )

            fk_motion = EasyDict(
                {k: v[0] if torch.is_tensor(v) else v for k, v in fk_motion.items()}
            )

            fk_motion.pose_aa = pose_aa
            res[curr_id] = fk_motion

        if queue is not None:
            queue.put(res)
        else:
            return res
            
    def get_motion_state_intervaled(
            self,
            motion_ids,
            motion_times,
            offset=None
    ):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(
            motion_times, motion_len, num_frames, dt
        )

        frame_idx = ((1.0 - blend) * frame_idx0 + blend * frame_idx1).astype(int)
        fl = frame_idx + self.length_starts[motion_ids]

        dof_pos = self.dof_pos[fl]
        body_vel = self.gvs[fl]
        body_ang_vel = self.gavs[fl]
        xpos = self.gts[fl, :]
        xquat = self.grs[fl]
        dof_vel = self.dvs[fl]
        qpos = self.qpos[fl]
        qvel = self.qvel[fl]

        if offset is not None:
            xpos = xpos + offset
            qpos = qpos.copy()
            qpos[..., :3] = qpos[..., :3] + offset

        return EasyDict(
            {
                "root_pos": xpos[..., 0, :].copy(),
                "root_rot": xquat[..., 0, :].copy(),
                "dof_pos": dof_pos.copy(),
                "root_vel": body_vel[..., 0, :].copy(),
                "root_ang_vel": body_ang_vel[..., 0, :].copy(),
                "dof_vel": dof_vel.reshape(dof_vel.shape[0], -1),
                "motion_aa": self._motion_aa[fl],
                "xpos": xpos,
                "xquat": xquat,
                "body_vel": body_vel,
                "body_ang_vel": body_ang_vel,
                # "motion_bodies": self._motion_bodies[motion_ids],
                "qpos": qpos,
                "qvel": qvel,
            }
        )

    def get_motion_length(self, motion_ids=None):
        if motion_ids is None:
            return self._motion_lengths
        else:
            return self._motion_lengths[motion_ids]

    def num_all_motions(self) -> int:
        """
        Returns the total number of motions in the dataset.

        Args:
            None

        Returns:
            The total number of motions in the dataset.
        """
        return self._num_unique_motions
    
    def fix_trans_height(
            self,
            pose_aa: torch.Tensor,
            trans: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        with torch.no_grad():
            frame_check = 30
            mesh_parser = self.fk_model.smpl_parser
            vertices_curr, _ = mesh_parser.get_joints_verts(
                pose_aa[:frame_check], th_trans=trans[:frame_check]
            )

            diff_fix = vertices_curr[:frame_check, ..., -1].min(dim=-1).values.min()

            trans[..., -1] -= diff_fix

            return trans, diff_fix
        
    def _calc_frame_blend(self, time, len, num_frames, dt):
        time = time.copy()
        phase = time / len
        phase = np.clip(phase, 0.0, 1.0)  # clip time to be within motion length.
        time[time < 0] = 0
        frame_idx0 = phase * (num_frames - 1)
        frame_idx1 = np.minimum(frame_idx0 + 1, num_frames - 1)

        blend = np.clip(
            (time - frame_idx0 * dt) / dt, 0.0, 1.0
        )  # clip blend to be within 0 and 1
        return frame_idx0, frame_idx1, blend
    
    def sample_motions(self, n=1):
        # breakpoint()
        motion_ids = np.random.choice(
            np.arange(len(self._curr_motion_ids)),
            size=n,
            p=self._sampling_batch_prob,
            replace=True,
        )
        return motion_ids