import os
import logging
import random
import torch.nn.functional as F
import hydra
import numpy as np
from tqdm import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf
import torch
import matplotlib.pyplot as plt
from src.aloha.aloha_scripts.visualize_episodes import STATE_NAMES,visualize_joints,load_hdf5

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
     "add", lambda *numbers: sum(numbers)
)
torch.cuda.empty_cache()


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def plot_joint_positions(pred_actions, true_actions, idx, save_dir):
    timesteps = np.arange(true_actions.shape[0])
    fig, axes = plt.subplots(5, 3, figsize=(20, 25))
    axes = axes.flatten()
    all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    for i in range(14):
        ax = axes[i]
        ax.plot(timesteps, true_actions[:, i], label='True', color='blue')
        ax.plot(timesteps, pred_actions[:, i], label='Predicted', color='red', linestyle='--')
        ax.set_title(f'Joint {i}: {all_names[i]}')
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Position')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'joint_positions_{idx}.png'))
    plt.close()

@hydra.main(config_path="configs", config_name="aloha_robot_config.yaml",version_base=None)
def main(cfg: DictConfig) -> None:
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.group,
        mode="disabled",
        config=wandb.config
    )
    ckpt_path = '/mnt/d/kit/ALR/dataset/test_david'
    save_dir = os.path.join(ckpt_path, 'test_results')
    os.makedirs(save_dir, exist_ok=True)

    total_loss = 0
    num_episodes = 0

    agent = hydra.utils.instantiate(cfg.agents)
    agent.load_pretrained_model(ckpt_path, sv_name=agent.eval_model_name)

    test_dataloader = agent.test_dataloader
    dataset_dir = cfg.train_data_path
    dataset_name = 'episode_0'
    _, _, action, _ = load_hdf5(dataset_dir, dataset_name)
    episode_length = 735
    with torch.no_grad():
        pred_full_traj = []
        true_full_traj = []
        for idx, data in enumerate(tqdm(test_dataloader), 0):
            images, true_actions = data
            images = {k: v.cpu().numpy() for k, v in images.items()}
            # true_actions = true_actions.cpu().numpy()
            predicted_action = agent.predict(images)
            pred_full_traj.extend(predicted_action)
            # true_full_traj.extend(true_actions)
            
            # if len(pred_full_traj) % episode_length ==0:
                
        
            #     pred_full_traj = torch.tensor(pred_full_traj, dtype=torch.float32).to(agent.device)
            #     true_full_traj = torch.tensor(true_full_traj, dtype=torch.float32).to(agent.device)
            #     pred_full_traj = agent.scaler.inverse_scale_output(pred_full_traj)
            #     true_full_traj = agent.scaler.inverse_scale_output(true_full_traj)
            #     pred_full_traj = pred_full_traj.cpu().numpy()
            #     true_full_traj = true_full_traj.cpu().numpy()

            #     plot_joint_positions(pred_full_traj, true_full_traj, num_episodes, save_dir)
            #     loss = F.mse_loss(torch.tensor(pred_full_traj), torch.tensor(true_full_traj))
            #     print(f"MSE Loss: {loss:.4f}")
            #     total_loss += loss.item()
            #     num_episodes += 1
            #     pred_full_traj = []
            #     true_full_traj = []

            # 当 idx 是 750 的倍数时，处理并保存轨迹
            if len(pred_full_traj) %episode_length == 0:
                # 处理当前 episode 的轨迹段
                # segment_length = min(episode_length, len(pred_full_traj))
                # pred_segment = np.array(pred_full_traj[:segment_length])
                
                pred_segment = torch.tensor(pred_full_traj, dtype=torch.float32).to(agent.device)
                pred_segment = agent.scaler.inverse_scale_output(pred_segment)
                pred_segment = pred_segment.cpu().numpy()
                
                visualize_joints(action[:episode_length], pred_segment, plot_path=os.path.join(save_dir, f'episode_{num_episodes}_qpos.png'))

                # 计算损失
                loss = F.mse_loss(torch.tensor(pred_segment), torch.tensor(action[:episode_length]))
                total_loss += loss.item()
                num_episodes += 1

                print(f"Episode {num_episodes}, MSE Loss: {loss:.4f}")

                # 重置 `pred_full_traj` 为未处理部分或清空
                pred_full_traj = []
                if num_episodes < 10:  # total_episodes 是你定义的总 episode 数量
                    _, _, action, _ = load_hdf5(dataset_dir, f'episode_{num_episodes}')

        avg_loss = total_loss / num_episodes
        print(f"Testing completed. Results saved in: {save_dir}")
        print(f"Average MSE Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()
