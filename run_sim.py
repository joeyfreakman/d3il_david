import os
import logging

import hydra
import numpy as np

import wandb
from omegaconf import DictConfig, OmegaConf
import torch
from agents.utils.sim_path import sim_framework_path


log = logging.getLogger(__name__)


OmegaConf.register_new_resolver(
     "add", lambda *numbers: sum(numbers)
)
torch.cuda.empty_cache()


# @hydra.main(config_path="/home/alr_admin/atalay/d3il_david/logs/real_robot_pickPlacing/sweeps/oc_ddpm/2024-06-02/22-52-33/task_suite=pickPlacing/.hydra", config_name="config.yaml")
@hydra.main(config_path="/home/alr_admin/atalay/d3il_david/weights/6_6/task_suite=cupStacking/.hydra", config_name="config.yaml")
def main(cfg: DictConfig) -> None:

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # init wandb logger and config from hydra path
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode="disabled",
        config=wandb.config
    )

    cfg.task_suite = 'cupStacking'
    cfg.if_sim = True
    agent = hydra.utils.instantiate(cfg.agents)
    agent.load_pretrained_model("/home/alr_admin/atalay/d3il_david/weights/6_6/task_suite=cupStacking",
                                sv_name='last_ddpm.pth')

    env_sim = hydra.utils.instantiate(cfg.simulation)
    env_sim.test_agent(agent)

    log.info("done")

    wandb.finish()


if __name__ == "__main__":
    main()