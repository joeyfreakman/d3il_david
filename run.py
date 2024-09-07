import os
import logging
import random

import hydra
import numpy as np

import wandb
from omegaconf import DictConfig, OmegaConf
import torch


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


@hydra.main(config_path="configs", config_name="aloha_robot_config.yaml",version_base=None)
def main(cfg: DictConfig) -> None:

    # if cfg.seed in [0, 1]:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # elif cfg.seed in [2, 3]:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # elif cfg.seed in [4, 5]:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    set_seed_everywhere(cfg.seed)

    # init wandb logger and config from hydra path
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.group,
        mode="online",
        config=wandb.config
    )
    agent = hydra.utils.instantiate(cfg.agents)
    if cfg.load_checkpoint:
        checkpoint_path = cfg.checkpoint_dir
        checkpoint_name = cfg.checkpoint_name
        if os.path.exists(os.path.join(checkpoint_path, checkpoint_name)):
            agent.load_pretrained_model(checkpoint_path, sv_name=checkpoint_name)
            log.info(f"Loaded pretrained model from {os.path.join(checkpoint_path, checkpoint_name)}")
        else:
            log.warning(f"Checkpoint {os.path.join(checkpoint_path, checkpoint_name)} not found. Starting from scratch.")
    # train the agent
    agent.train_vision_agent()

    # load the model performs best on the evaluation set
    # agent.load_pretrained_model(agent.working_dir, sv_name=agent.eval_model_name)

    # simulate the model
    # env_sim = hydra.utils.instantiate(cfg.simulation)
    # env_sim.test_agent(agent)

    log.info("done")

    wandb.finish()


if __name__ == "__main__":
    main()