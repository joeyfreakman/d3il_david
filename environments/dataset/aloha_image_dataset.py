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
import torch.nn.functional as F
import argparse
import torch.utils.data
class Aloha_Image_dataset(TrajectoryDataset):
    def __init__(
            self,
            data_directory: os.PathLike,
            task_name: str,
            obs_dim:14,
            action_dim:14,
            max_len_data: int = 750,
            device="cpu",
            window_size: int = 1,
    ):
        task_config = TASK_CONFIGS[task_name]
        num_episodes = task_config['num_episodes']
        self.episode_len = task_config['episode_len']
        self.camera_names = task_config['camera_names']

        super().__init__(
            data_directory=data_directory,
            device=device,
            obs_dim=obs_dim,  # 14 (qpos) 
            action_dim=action_dim,  # 14 
            max_len_data=max_len_data,
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
        try: 
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

                
                action = torch.from_numpy(root["/action"][start:end]).float()

                return images, action
        except Exception as e:
            print(f"Error loading episode {episode_id}: {e}")
            


    def get_seq_length(self, idx):
        with h5py.File(self.episodes[idx], "r") as root:
            return root['/action'].shape[0]

    def get_all_actions(self):
        all_actions = []
        for episode in self.episodes:
            with h5py.File(episode, "r") as root:
                all_actions.append(torch.from_numpy(root["/action"][()]).float())
        return torch.cat(all_actions, dim=0)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_directory", type=str, required=True)
#     parser.add_argument("--task_name", type=str, required=True)
#     args = parser.parse_args()

#     dataset = Aloha_Image_dataset(
#         data_directory=Path(args.data_directory),
#         task_name=args.task_name,
#         obs_dim=14,
#         action_dim=14,
#         max_len_data=750,
#         device="cpu",
#         window_size=1,
#     )

#     train_data_loader = torch.utils.data.DataLoader(dataset=dataset,
#                                                     batch_size=1,
#                                                     shuffle=True,
#                                                     num_workers=0,
#                                                     pin_memory=False,
#                                                     )
#     try:
#         for idx, data in enumerate(train_data_loader):
#             images, action, mask = data
#             print(f"Batch {idx}: images shape: {images['cam_high'].shape}, action shape: {action.shape}")
            
#     except Exception as e:
#         print(f"Error occurred: {e}")
#         import traceback
#         traceback.print_exc()

        
