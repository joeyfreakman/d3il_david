from collections import deque
import os
import logging 
from typing import Optional

from omegaconf import DictConfig
import hydra
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import wandb
import einops

from src.d3il_david.agents.base_agent import BaseAgent
from src.d3il_david.agents.models.oc_ddpm.ema import ExponentialMovingAverage

log = logging.getLogger(__name__)

class DiffusionPolicy(nn.Module):
    def __init__(self, 
                 model: DictConfig, 
                 obs_encoder: DictConfig,
                 device: str = "cpu"):
        super(DiffusionPolicy, self).__init__()

        self.camera_names = ['cam_high' , 'cam_left_wrist','cam_low', 'cam_right_wrist']
        self.obs_encoder = hydra.utils.instantiate(obs_encoder).to(device)
        self.model = hydra.utils.instantiate(model).to(device)

    def forward(self, inputs, goal, action=None, if_train=False):
        images = inputs
        B, T = next(iter(images.values())).shape[:2]
        obs_dict = {}

        for cam in self.camera_names:
            img = inputs[cam]
            B, T, C, H, W = img.size()
            img = img.view(B * T, C, H, W)
            obs_dict[cam] = img
        
        
        # Encode the observations using the encoder
        obs = self.obs_encoder(obs_dict)
        obs = obs.view(B, T, -1)

        if if_train:
            return self.model.loss(action, obs, goal)

        # make prediction
        pred = self.model(obs, goal)

        return pred

    def get_params(self):
        return self.parameters()
    
class AlohaImageDDPMAgent(BaseAgent):
    def __init__(
            self,
            model: DictConfig,
            optimization: DictConfig,
            trainset: DictConfig,
            valset: DictConfig,
            train_batch_size,
            val_batch_size,
            num_workers,
            device: str,
            epoch: int,
            scale_data,
            use_ema: bool,
            discount: int,
            decay: float,
            update_ema_every_n_steps: int,
            goal_window_size: int,
            window_size: int,
            obs_seq_len: int,
            action_seq_size: int,
            pred_last_action_only: bool = False,
            diffusion_kde: bool = False,
            diffusion_kde_samples: int = 100,
            goal_conditioned: bool = False,
            eval_every_n_epochs: int = 50
    ):
        super().__init__(model, trainset=trainset, valset=valset, train_batch_size=train_batch_size,
                         val_batch_size=val_batch_size, num_workers=num_workers, device=device,
                         epoch=epoch, scale_data=scale_data, eval_every_n_epochs=eval_every_n_epochs)
        
        # Define the bounds for the sampler class
        self.model.model.min_action = torch.from_numpy(self.scaler.y_bounds[0, :]).to(self.device)
        self.model.model.max_action = torch.from_numpy(self.scaler.y_bounds[1, :]).to(self.device)

        self.eval_model_name = "eval_best_ddpm.pth"
        self.last_model_name = "last_ddpm.pth"

        self.optimizer = hydra.utils.instantiate(
            optimization, params=self.model.parameters()
        )

        self.steps = 0

        self.ema_helper = ExponentialMovingAverage(self.model.parameters(), decay, self.device)
        self.use_ema = use_ema
        self.discount = discount
        self.decay = decay
        self.update_ema_every_n_steps = update_ema_every_n_steps
        # here all the parameters required for the GPT variant
        self.goal_window_size = goal_window_size
        self.window_size = window_size
        self.pred_last_action_only = pred_last_action_only

        self.goal_condition = goal_conditioned

        # if we use DiffusionGPT we need an action context
        if not self.pred_last_action_only:
            self.action_context = deque(maxlen=self.window_size - 1)
            self.que_actions = True
        else:
            self.que_actions = False

        self.diffusion_kde = diffusion_kde
        self.diffusion_kde_samples = diffusion_kde_samples

        self.obs_seq_len = obs_seq_len
        self.action_seq_size = action_seq_size
        self.action_counter = self.action_seq_size

        self.des_robot_pos_context = deque(maxlen=self.window_size)

        self.obs_context = deque(maxlen=self.obs_seq_len)
        self.goal_context = deque(maxlen=self.goal_window_size)
        self.camera_names = ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
        self.image_contexts = {cam: deque(maxlen=self.obs_seq_len) for cam in self.camera_names}

    def train_agent(self):

        for num_epoch in tqdm(range(self.epoch)):
            if not (num_epoch+1) % self.eval_every_n_epochs:
                test_mse = []
                for idx, data in enumerate(self.test_dataloade):
                    images, action = data
                    mean_mse = self.evaluate(images, action)
                    test_mse.append(mean_mse)
                avrg_test_mse = sum(test_mse) / len(test_mse)

                log.info("Epoch {}: Mean test mse is {}".format(num_epoch, avrg_test_mse))
                if avrg_test_mse < best_test_mse:
                    best_test_mse = avrg_test_mse
                    self.store_model_weights(self.working_dir, sv_name=self.eval_model_name)

                    wandb.log(
                        {
                            "best_model_epochs": num_epoch
                        }
                    )

                    log.info('New best test loss. Stored weights have been updated!')

            train_loss = []
            for idx, data in enumerate(self.train_dataloader):
                images,action = data
                batch_loss = self.train_step(images, action)
                train_loss.append(batch_loss)

                wandb.log({"loss": batch_loss})

        self.store_model_weights(self.working_dir, sv_name=self.last_model_name)
        log.info("Training done!")

    def train_step(self, images: dict, action: torch.Tensor) -> float:
        self.model.train()

        # process image data
        for cam in self.camera_names:
            images[cam] = images[cam].to(self.device).float() / 255.0

        action = action.to(self.device).float()

        # make sure to only use obs_seq_len length of observations
        for cam in self.camera_names:
            images[cam] = images[cam][:, :self.obs_seq_len]

        action = action[:, self.obs_seq_len-1:]
        state = images

        # compute loss
        loss = self.model(state, None, action=action, if_train=True)

        # backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1

        if self.steps % self.update_ema_every_n_steps == 0:
            self.ema_helper.update(self.model.parameters())

        return loss.item()

    @torch.no_grad()
    def evaluate(self, images: dict, action: torch.Tensor) -> float:
        # process image data
        for cam in self.camera_names:
            images[cam] = images[cam].to(self.device).float() / 255.0

        
        action = action.to(self.device).float()

        # makes sure to only use obs_seq_len length of observations
        for cam in self.camera_names:
            images[cam] = images[cam][:, :self.obs_seq_len]
        
        action = action[:, self.obs_seq_len-1:]

        state = images

        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())

        self.model.eval()

        loss = self.model.loss(action, state, None)

        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())

        return loss.mean().item()
    
    def reset(self):
        # Reset any necessary state variables
        self.action_counter = self.action_seq_size
        self.des_robot_pos_context.clear()
        self.obs_context.clear()
        self.goal_context.clear()
        for cam in self.camera_names:
            self.image_contexts[cam].clear()
        
        # If you're using action_context, reset it too
        if hasattr(self, 'action_context'):
            self.action_context.clear()

    @torch.no_grad()
    def predict(self, state: tuple, goal: Optional[torch.Tensor] = None, extra_args=None) -> torch.Tensor:
        images= state

        # process image data
        for cam in self.camera_names:
            img = torch.from_numpy(images[cam]).to(self.device).float().unsqueeze(0) / 255.0
            self.image_contexts[cam].append(img)

        if self.action_counter == self.action_seq_size:
            self.action_counter = 0

            if self.use_ema:
                self.ema_helper.store(self.model.parameters())
                self.ema_helper.copy_to(self.model.parameters())

            self.model.eval()

            input_images = {cam: torch.cat(list(self.image_contexts[cam]), dim=1) for cam in self.camera_names}
            
            input_state = input_images

            model_pred = self.model(input_state, goal)

            if self.use_ema:
                self.ema_helper.restore(self.model.parameters())

            self.curr_action_seq = model_pred

        next_action = self.curr_action_seq[:, self.action_counter, :]
        self.action_counter += 1
        return next_action.detach().cpu().numpy()