import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import re

# Joint List for the 32D Observation (16 angles + 16 velocities)
# 1-6: root (tx, ty, tz, tilt, list, rot)
# 7-9: hip_flexion_r, hip_adduction_r, hip_rotation_r
# 10-11: osl_knee_angle_r, osl_ankle_angle_r
# 12-14: hip_flexion_l, hip_adduction_l, hip_rotation_l
# 15-16: knee_angle_l, ankle_angle_l
JOINT_COLUMNS = [
    'pelvis_tx', 'pelvis_ty', 'pelvis_tz', 'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
    'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
    'osl_knee_angle_r', 'osl_ankle_angle_r',
    'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
    'knee_angle_l', 'ankle_angle_l'
]

# Mapping specifically for OSL model if column names in .mot differ
# (Based on provided header it matches for mostly everything)
MOT_TO_OBS = {
    'knee_angle_r': 'osl_knee_angle_r',
    'ankle_angle_r': 'osl_ankle_angle_r',
}

def extract_speed(filename):
    """Extracts speed from filename like tf01_0p6_01_rotated_ik.mot"""
    match = re.search(r'(\d+)p(\d+)', filename)
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")
    return 0.0

def parse_mot(filepath):
    """Parses .mot file and returns a dataframe + metadata."""
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
            
    df = pd.read_csv(filepath, sep='\t', skiprows=header_end)
    if len(df.columns) == 1:
        # Retry with space separator
        df = pd.read_csv(filepath, sep='\s+', skiprows=header_end)
        
    return df, in_degrees

def generate_trajectories(data_dir, output_path):
    mot_files = list(Path(data_dir).glob("*.mot"))
    print(f"Found {len(mot_files)} .mot files.")
    
    trajectories = []
    
    for mot_file in tqdm(mot_files):
        filename = mot_file.name
        if filename == "processed_motions.joblib": continue
        
        speed = extract_speed(filename)
        df, in_degrees = parse_mot(str(mot_file))
        
        # Apply name mapping
        df = df.rename(columns=MOT_TO_OBS)
        
        num_frames = len(df)
        times = df['time'].values
        dt = times[1] - times[0] if num_frames > 1 else 0.033
        
        unit_scale = np.pi / 180.0 if in_degrees else 1.0
        
        # 1. Extract Angles
        angles = np.zeros((num_frames, len(JOINT_COLUMNS)), dtype=np.float32)
        for i, col in enumerate(JOINT_COLUMNS):
            if col in df.columns:
                # Root translations shouldn't be scaled by degrees-to-rad
                scale = 1.0 if 'pelvis_t' in col else unit_scale
                angles[:, i] = df[col].values * scale
            else:
                # Handle missing columns if any
                print(f"Warning: Column {col} not found in {filename}")
        
        # 2. Compute Velocities (Finite Difference)
        velocities = np.zeros_like(angles)
        velocities[:-1] = np.diff(angles, axis=0) / dt
        velocities[-1] = velocities[-2] # Pad last frame
        
        # 3. Concatenate into Observation (32D)
        obs_traj = np.concatenate([angles, velocities], axis=1) # Shape: (T, 32)
            
        trajectories.append({
            'speed': speed,
            'observation': torch.from_numpy(obs_traj)
        })

    # Save all trajectories
    print(f"Saving {len(trajectories)} trajectories to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(trajectories, output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/media/tripan/Data/DDP/amputee_data/training_data")
    parser.add_argument("--output_path", type=str, default="data/expert_trajectories.pth")
    args = parser.parse_args()
    
    generate_trajectories(args.data_dir, args.output_path)
