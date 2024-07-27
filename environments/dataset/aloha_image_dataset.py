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
import matplotlib.pyplot as plt
class Aloha_Image_dataset(TrajectoryDataset):
    def __init__(
            self,
            data_directory: os.PathLike,
            task_name: str,
            obs_dim:14,
            action_dim:14,
            max_len_data: int = 750,
            device="cuda",
            window_size: int = 1,
            obs_seq_len: int = 2,
    ):
        task_config = TASK_CONFIGS[task_name]
        num_episodes = task_config['num_episodes']
        self.episode_len = task_config['episode_len']
        self.camera_names = task_config['camera_names']
        self.obs_seq_len = obs_seq_len

        super().__init__(
            data_directory=data_directory,
            device=device,
            obs_dim=obs_dim,  # 14 (qpos) 
            action_dim=action_dim,  # 14 
            max_len_data=max_len_data,
            window_size=window_size,
            obs_seq_len=obs_seq_len,
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
                    img_data = root[f"/observations/images/{cam_name}"][start:start+self.obs_seq_len]
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
    
    def visualize_grouped_action_trajectory(self, action):
        T = action.shape[0]
        time = range(T)
        
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        fig.suptitle('Grouped Action Trajectory')
        
        joint_groups = [
            ('Left Arm', [0, 1, 2, 3, 4, 5 ]),
            ('Right Arm', [7, 8, 9, 10, 11, 12 ]),
            ('left_gripper',[6]),
            ('right_gripper',[13])
            
        ]
        
        for i, (group_name, joint_indices) in enumerate(joint_groups):
            row = i // 2
            col = i % 2
            for j in joint_indices:
                axs[row, col].plot(time, action[:, j], label=f'Joint {j+1}')
            axs[row, col].set_title(group_name)
            axs[row, col].set_xlabel('Time steps')
            axs[row, col].set_ylabel('Joint angles')
            axs[row, col].legend()
            axs[row, col].grid(True)
        
        plt.tight_layout()
        plt.show()
        

# $python aloha_image_dataset.py --data_directory /mnt/d/kit/ALR/dataset/ttp_compressed --task_name test
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_directory", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    args = parser.parse_args()

    dataset = Aloha_Image_dataset(
        data_directory=Path(args.data_directory),
        task_name=args.task_name,
        obs_dim=14,
        action_dim=14,
        max_len_data=750,
        device="cuda",
        window_size=16,
    )
    idx = np.random.randint(0, len(dataset))
    images, action = dataset[idx]
    
    if images is not None and action is not None:
        output_dir = os.path.join(dataset.data_directory, "plot_david")
        os.makedirs(output_dir, exist_ok=True)
        dataset.visualize_grouped_action_trajectory(action)
        plt.savefig(os.path.join(output_dir, f"action.png"))
        for t in range(dataset.window_size):
            plt.figure(figsize=(10, 5))
            for cam_idx, cam_name in enumerate(dataset.camera_names):
                plt.subplot(1, len(dataset.camera_names), cam_idx + 1)
                img_rgb = cv2.cvtColor(
                    images[cam_name][t].permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB
                )
                plt.imshow(img_rgb)
                plt.title(f"{cam_name} at timestep {t}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"images_timestep_{t}.png"))
            print(f"Saved images_timestep_{t}.png")
            plt.close()


    # train_data_loader = torch.utils.data.DataLoader(dataset=dataset,
    #                                                 batch_size=1,
    #                                                 shuffle=True,
    #                                                 num_workers=4,
    #                                                 pin_memory=True,
    #                                                 prefetch_factor=10
    #                                                 )
    # try:
    #     for idx, data in enumerate(train_data_loader):
    #         images, action = data
            
            
            
    # except Exception as e:
    #     print(f"Error occurred: {e}")
    #     import traceback
    #     traceback.print_exc()

        
