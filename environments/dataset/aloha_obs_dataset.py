import logging
from src.config.dataset_config import TASK_CONFIGS
from src.d3il_david.environments.dataset.base_dataset import TrajectoryDataset
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
import h5py

class Aloha_dataset(TrajectoryDataset):
    def __init__(
            self,
            task_name: str,
            device="cpu",
            window_size: int = 1,
    ):
        task_config = TASK_CONFIGS[task_name]
        data_directory = task_config['dataset_dir']
        num_episodes = task_config['num_episodes']
        self.episode_len = task_config['episode_len']
        self.camera_names = task_config['camera_names']

        super().__init__(
            data_directory=data_directory,
            device=device,
            obs_dim=14,  # 14 (qpos) 
            action_dim=14,
            max_len_data=self.episode_len,
            window_size=window_size,
        )

        logging.info(f"Loading Real Robot Dataset for task: {task_name}")

        self.episodes = []

        for episode_id in tqdm(range(num_episodes)):
            dataset_path = os.path.join(data_directory, f"episode_{episode_id}.hdf5")
            if os.path.exists(dataset_path):
                self.episodes.append(dataset_path)

        self.num_data = len(self.episodes)
        self.slices = self.get_slices()

    def get_slices(self):
        slices = []
        for episode_id in range(self.num_data):
            with h5py.File(self.episodes[episode_id], "r") as root:
                T = min(root['/action'].shape[0], self.episode_len)
                if T - self.window_size < 0:
                    print(f"Ignored short sequence #{episode_id}: len={T}, window={self.window_size}")
                else:
                    slices += [(episode_id, start, start + self.window_size)
                               for start in range(T - self.window_size + 1)]
        return slices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        episode_id, start, end = self.slices[idx]
        
        with h5py.File(self.episodes[episode_id], "r") as root:
            compressed = root.attrs.get("compress", False)
            
            # make sure the sequence is within the episode length
            end = min(end, start + self.episode_len)
            
            images = {}
            for cam_name in self.camera_names:
                img_data = root[f"/observations/images/{cam_name}"][start:end]
                if compressed:
                    img_data = np.array([cv2.imdecode(frame, 1) for frame in img_data])
                img_data = torch.from_numpy(img_data).float() / 255.0
                img_data = img_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
                images[cam_name] = img_data

            qpos = torch.from_numpy(root["/observations/qpos"][start:end]).float()
            action = torch.from_numpy(root["/action"][start:end]).float()

        # if the sequence is shorter than the episode length, pad it
        seq_len = end - start
        if seq_len < self.episode_len:
            pad_len = self.episode_len - seq_len
            for cam_name in images:
                images[cam_name] = torch.pad(images[cam_name], (0, 0, 0, 0, 0, 0, 0, pad_len))
            qpos = torch.pad(qpos, (0, 0, 0, pad_len))
            action = torch.pad(action, (0, 0, 0, pad_len))

        # create a mask to indicate the valid length of the sequence
        mask = torch.ones(self.episode_len, dtype=torch.bool)
        mask[seq_len:] = False

        return images, qpos, action, mask


    def get_seq_length(self, idx):
        with h5py.File(self.episodes[idx], "r") as root:
            return root['/action'].shape[0]

    def get_all_actions(self):
        all_actions = []
        for episode in self.episodes:
            with h5py.File(episode, "r") as root:
                all_actions.append(torch.from_numpy(root["/action"][()]).float())
        return torch.cat(all_actions, dim=0)
