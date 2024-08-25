import os
import logging
from einops import rearrange
import hydra
import numpy as np
import cv2
import wandb
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import torch
from src.aloha.aloha_scripts.constants import DT, PUPPET_GRIPPER_JOINT_OPEN
from src.config.dataset_config import TASK_CONFIGS, DATA_DIR
from src.aloha.aloha_scripts.real_env import make_real_env
from src.aloha.aloha_scripts.robot_utils import move_grippers
from src.aloha.aloha_scripts.visualize_episodes import save_videos,load_hdf5,STATE_NAMES




log = logging.getLogger(__name__)


OmegaConf.register_new_resolver(
     "add", lambda *numbers: sum(numbers)
)
torch.cuda.empty_cache()


@hydra.main(config_path="configs", config_name="aloha_robot_config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode="online",
        config=wandb.config
    )

    agent = hydra.utils.instantiate(cfg.agents)
    if cfg.load_checkpoint:
        checkpoint_path = cfg.checkpoint_dir
        checkpoint_name = cfg.checkpoint_name
        ckpt_dir = os.path.join(checkpoint_path, checkpoint_name)
        if os.path.exists(ckpt_dir):
            agent.load_pretrained_model(checkpoint_path, sv_name=checkpoint_name)
        else:
            log.warning(f"Checkpoint {ckpt_dir} not found. Starting from scratch.")

    env = make_real_env(init_node=True)
    env_max_reward = 0  # You might want to set this based on your task
    onscreen_render = cfg.onscreen_render
    save_episode = cfg.save_episode
    max_timesteps = cfg.max_len_data
    camera_names = cfg.trainset.camera_names
    onscreen_cam = 'cam_high'

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []

    n_existing_rollouts = (
        len([f for f in os.listdir(checkpoint_path) if f.startswith("video")])
        if save_episode
        else 0
    )
    print(f"{n_existing_rollouts=}")

    for rollout_id in range(num_rollouts):
        ts = env.reset()
        agent.action_counter = 0
        if onscreen_render:
            fig, ax = plt.subplots()
            plt_img = ax.imshow(env.render(camera_id=onscreen_cam))
            plt.ion()

        image_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                if onscreen_render:
                    image = env.render(camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                obs = ts.observation
                if "images" in obs:
                    image_list.append(obs["images"])
                else:
                    image_list.append({"main": obs["image"]})

                curr_image = get_image(ts, camera_names, save_dir=checkpoint_path if t == 0 else None)
                predicted_action = agent.predict(curr_image)  # Shape: (1, 14)
                
                raw_action = predicted_action.squeeze(0).cpu().numpy()  # Shape: (14,)
                action = agent.scaler.inverse_scale_output(raw_action)
                # if np.any(np.abs(action) > 0.1):
                target_qpos = action

                ts = env.step(target_qpos)

                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()

        move_grippers(
            [env.puppet_bot_left, env.puppet_bot_right],
            [PUPPET_GRIPPER_JOINT_OPEN] * 2,
            move_time=0.5,
        )

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards != None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(
            f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}"
        )
        if run:
            wandb.log(
                {
                    "test/episode_return": episode_return,
                    "test/episode_highest_reward": episode_highest_reward,
                    "test/env_max_reward": env_max_reward,
                    "test/success": episode_highest_reward == env_max_reward,
                },
                step=rollout_id,
            )

        if save_episode:
            video_name = f"video{rollout_id+n_existing_rollouts}.mp4"
            save_videos(
                image_list,
                DT,
                video_path=os.path.join(checkpoint_path, video_name),
                cam_names=camera_names,
            )
            if run:
                wandb.log(
                    {
                        "test/video": wandb.Video(
                            os.path.join(checkpoint_path, f"video{rollout_id}.mp4"),
                            fps=50,
                            format="mp4",
                        )
                    },
                    step=rollout_id,
                )

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
    for r in range(env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f"Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n"

    print(summary_str)

    result_file_name = "result_" + checkpoint_name.split(".")[0] + ".txt"
    with open(os.path.join(checkpoint_path, result_file_name), "w") as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write("\n\n")
        f.write(repr(highest_rewards))

    if run:
        wandb.log({"test/success_rate": success_rate, "test/avg_return": avg_return})

    log.info("done")
    wandb.finish()


def get_image(ts, camera_names, save_dir=None, t=None):
    curr_images = {}
    for cam_name in camera_names:
        curr_image = ts.observation["images"][cam_name]
        curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)
        curr_image = rearrange(curr_image, "h w c -> c h w")
        curr_images[cam_name] = curr_image

    if save_dir is not None:
        concat_images = [rearrange(img, "c h w -> h w c") for img in curr_images.values()]
        concat_image = np.concatenate(concat_images, axis=1)
        concat_image = cv2.cvtColor(concat_image, cv2.COLOR_RGB2BGR)
        img_name = "init_visualize.png" if t is None else f"gpt/{t=}.png"
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, (concat_image * 255).astype(np.uint8))

    return curr_images

if __name__ == "__main__":
    main()