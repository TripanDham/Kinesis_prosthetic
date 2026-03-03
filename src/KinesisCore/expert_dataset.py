import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ExpertDataset(Dataset):
    """
    Dataset for loading expert trajectories and providing temporal history windows.
    Each trajectory is a dictionary with 'speed' and 'observation' (torch.Tensor of shape (T, ObsDim)).
    """
    def __init__(self, path, history_len=3):
        self.trajectories = torch.load(path)
        self.history_len = history_len
        
        # Build index of valid windows (don't cross trajectory boundaries)
        self.valid_indices = []
        for traj_idx, traj in enumerate(self.trajectories):
            obs = traj['observation']
            num_frames = obs.size(0)
            if num_frames >= history_len:
                for frame_idx in range(history_len - 1, num_frames):
                    self.valid_indices.append((traj_idx, frame_idx))
        
        print(f"ExpertDataset initialized with {len(self.valid_indices)} samples from {len(self.trajectories)} trajectories.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        traj_idx, frame_idx = self.valid_indices[idx]
        obs = self.trajectories[traj_idx]['observation']
        
        # Extract window [frame_idx - history_len + 1, ..., frame_idx]
        window = obs[frame_idx - self.history_len + 1 : frame_idx + 1]
        
        # Concatenate temporal history into a single vector
        # Shape: (HistoryLen, ObsDim) -> (HistoryLen * ObsDim,)
        return window.reshape(-1)

def get_expert_loader(path, batch_size, history_len=3, shuffle=True):
    dataset = ExpertDataset(path, history_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
