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

        self.camera_names = ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
        self.obs_encoder = nn.ModuleDict({
            cam: hydra.utils.instantiate(obs_encoder).to(device) for cam in self.camera_names
        })
        
        
        # Adjust the input size of the model based on your encoders' output sizes
        encoded_dim = len(self.camera_names) * obs_encoder.rgb_model.output_size + 14  # 14 for qpos
        self.model = hydra.utils.instantiate(model, condition_dim=encoded_dim).to(device)

    def forward(self, inputs, goal, action=None, if_train=False):
        images, qpos = inputs
        
        B, T = qpos.shape[:2]
        
        # Encode images from each camera
        encoded_images = []
        for cam in self.camera_names:
            img = images[cam].view(B * T, *images[cam].shape[2:])
            encoded = self.obs_encoder[cam](img)
            encoded = encoded.view(B, T, -1)
            encoded_images.append(encoded)
        
        
        # Concatenate all encodings
        obs = torch.cat(encoded_images + [qpos], dim=-1)

        if if_train:
            return self.model.loss(action, obs, goal)

        # make prediction
        pred = self.model(obs, goal)

        return pred

    def get_params(self):
        return self.parameters()
    

class AlohaDDPMAgent(BaseAgent):
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
        self.qpos_context = deque(maxlen=self.obs_seq_len)

    def train_agent(self):
        best_test_mse = 1e10

        for num_epoch in tqdm(range(self.epoch)):
            if not (num_epoch+1) % self.eval_every_n_epochs:
                test_mse = []
                for data in self.test_dataloader:
                    images, qpos, action, mask = data
                    mean_mse = self.evaluate(images, qpos, action)
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
            for data in self.train_dataloader:
                images, qpos, action, mask = data
                batch_loss = self.train_step(images, qpos, action)
                train_loss.append(batch_loss)

                wandb.log({"loss": batch_loss})

        self.store_model_weights(self.working_dir, sv_name=self.last_model_name)
        log.info("Training done!")

    def train_step(self, images: dict, qpos: torch.Tensor, action: torch.Tensor) -> float:
        self.model.train()

        # process image data
        for cam in self.camera_names:
            images[cam] = images[cam].to(self.device).float() / 255.0

        qpos = qpos.to(self.device).float()
        
        action = action.to(self.device).float()

        # make sure to only use obs_seq_len length of observations
        for cam in self.camera_names:
            images[cam] = images[cam][:, :self.obs_seq_len]
        qpos = qpos[:, :self.obs_seq_len]
        action = action[:, self.obs_seq_len-1:]

        state = (images, qpos)

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
    def evaluate(self, images: dict, qpos: torch.Tensor, action: torch.Tensor) -> float:
        # process image data
        for cam in self.camera_names:
            images[cam] = images[cam].to(self.device).float() / 255.0

        qpos = qpos.to(self.device).float()
        action = action.to(self.device).float()

        # makes sure to only use obs_seq_len length of observations
        for cam in self.camera_names:
            images[cam] = images[cam][:, :self.obs_seq_len]
        qpos = qpos[:, :self.obs_seq_len]
        action = action[:, self.obs_seq_len-1:]

        state = (images, qpos)

        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())

        self.model.eval()

        loss = self.model.loss(action, state, None)

        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())

        return loss.mean().item()

    @torch.no_grad()
    def predict(self, state: tuple, goal: Optional[torch.Tensor] = None, extra_args=None) -> torch.Tensor:
        images, qpos= state

        # process image data
        for cam in self.camera_names:
            img = torch.from_numpy(images[cam]).to(self.device).float().unsqueeze(0) / 255.0
            self.image_contexts[cam].append(img)

        qpos = torch.from_numpy(qpos).to(self.device).float().unsqueeze(0)
         
        self.qpos_context.append(qpos)

        if self.action_counter == self.action_seq_size:
            self.action_counter = 0

            if self.use_ema:
                self.ema_helper.store(self.model.parameters())
                self.ema_helper.copy_to(self.model.parameters())

            self.model.eval()

            input_images = {cam: torch.cat(list(self.image_contexts[cam]), dim=1) for cam in self.camera_names}
            input_qpos = torch.cat(list(self.qpos_context), dim=1)
            

            input_state = (input_images, input_qpos)

            model_pred = self.model(input_state, goal)

            if self.use_ema:
                self.ema_helper.restore(self.model.parameters())

            self.curr_action_seq = model_pred

        next_action = self.curr_action_seq[:, self.action_counter, :]
        self.action_counter += 1
        return next_action.detach().cpu().numpy()