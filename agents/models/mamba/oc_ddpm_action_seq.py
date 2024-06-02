import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import einops
import wandb
import os
from typing import Optional
import numpy as np
import logging
from agents.models.oc_ddpm.ema import ExponentialMovingAverage
from agents.base_agent import BaseAgent
from collections import deque
from tqdm import tqdm
from .utils import SinusoidalPosEmb
from .utils import (cosine_beta_schedule,
                    linear_beta_schedule,
                    vp_beta_schedule,
                    extract,
                    Losses
                    )

log = logging.getLogger(__name__)

class DiffusionEncDec(nn.Module):
    """Diffusion model with transformer architecture for state, goal, time and action tokens,
    with a context size of block_size"""

    def __init__(
            self,
            mamba: DictConfig,
            decoder: DictConfig,
            state_dim: int,
            action_dim: int,
            device: str,
            goal_conditioned: bool,
            embed_dim: int,
            embed_pdrob: float,
            goal_seq_len: int,
            obs_seq_len: int,
            action_seq_len: int,
            goal_drop: float = 0.1,
            linear_output: bool = False,
    ):
        super().__init__()

        self.mamba = hydra.utils.instantiate(mamba)
        self.decoder = hydra.utils.instantiate(decoder)

        self.device = device
        self.goal_conditioned = goal_conditioned
        if not goal_conditioned:
            goal_seq_len = 0
        # input embedding stem
        # first we need to define the maximum block size
        # it consists of the goal sequence length plus 1 for the sigma embedding and 2 the obs seq len
        block_size = goal_seq_len + action_seq_len + obs_seq_len + 1
        # the seq_size is a little different since we have state action pairs for every timestep
        seq_size = goal_seq_len + obs_seq_len - 1 + action_seq_len

        self.tok_emb = nn.Linear(state_dim, embed_dim)
        self.tok_emb.to(self.device)

        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrob)
        self.drop.to(self.device)

        # needed for calssifier guidance learning
        self.cond_mask_prob = goal_drop

        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.embed_dim = embed_dim

        self.block_size = block_size
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len

        # we need another embedding for the time
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.time_emb.to(self.device)
        # get an action embedding
        self.action_emb = nn.Linear(action_dim, embed_dim)
        self.action_emb.to(self.device)
        # action pred module
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100),
                nn.GELU(),
                nn.Linear(100, self.action_dim)
            )
        # self.action_pred = nn.Linear(embed_dim, action_dim) # less parameters, worse reward
        self.action_pred.to(self.device)

        self.apply(self._init_weights)

        log.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
                len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
                len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    # x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, g: torch.Tensor
    # def forward(self, x, t, state, goal):
    def forward(
            self,
            actions,
            time,
            states,
            goals,
            uncond: Optional[bool] = False,
            keep_last_actions: Optional[bool] = False
    ):

        # actions = actions[:, self.obs_seq_len-1:, :]
        # states = states[:, :self.obs_seq_len, :]

        if len(states.size()) != 3:
            states = states.unsqueeze(0)

        if len(actions.size()) != 3:
            actions = actions.unsqueeze(0)

        b, t, dim = states.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # get the time embedding
        times = einops.rearrange(time, 'b -> b 1')
        emb_t = self.time_emb(times)

        if self.goal_conditioned:

            if self.training:
                goals = self.mask_cond(goals)
            # we want to use unconditional sampling during clasisfier free guidance
            if uncond:
                goals = torch.zeros_like(goals).to(self.device)

            goal_embed = self.tok_emb(goals)

        # embed them into linear representations for the transformer
        state_embed = self.tok_emb(states)
        action_embed = self.action_emb(actions)

        position_embeddings = self.pos_emb[:, :(t + self.goal_seq_len + self.action_seq_len - 1), :]
        # note, that the goal states are at the beginning of the sequence since they are available
        # for all states s_1, ..., s_t otherwise the masking would not make sense
        if self.goal_conditioned:
            goal_x = self.drop(goal_embed + position_embeddings[:, :self.goal_seq_len, :])

        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len + t), :])
        action_x = self.drop(action_embed + position_embeddings[:, self.goal_seq_len:(t + self.goal_seq_len + self.action_seq_len - 1), :])

        if t > 1:
            state = state_x[:, :(t-1), :]
            past_action = action_x[:, :(t-1), :]
            action_pred = action_x[:, (t-1):, :]

            sa_seq = torch.stack([state, past_action], dim=1).permute(0, 2, 1, 3).reshape(b, 2 * (t - 1),self.embed_dim)
            current_state = state_x[:, -1, :].unsqueeze(1)
            # emb_t, s_0, a_0, ..., s_(t-1), a_(t-1), s_t
            input_seq = torch.cat([emb_t, sa_seq, current_state], dim=1)
        else:
            state = state_x
            action_pred = action_x
            input_seq = torch.cat([emb_t, state], dim=1)

        # encode the state, goal and latent z into the hidden dim
        mamba_output = self.mamba(input_seq)

        time_e = mamba_output[:, 0, :].unsqueeze(1)
        cur_s = mamba_output[:, -1, :].unsqueeze(1)

        cross_input = torch.cat([time_e, cur_s], dim=1)

        decoder_output = self.decoder(action_pred, cross_input)

        pred_actions = self.action_pred(decoder_output)

        return pred_actions

    def get_params(self):
        return self.parameters()


class DiffusionTwoMamba(nn.Module):
    """Diffusion model with transformer architecture for state, goal, time and action tokens,
    with a context size of block_size"""

    def __init__(
            self,
            mamba: DictConfig,
            decoder: DictConfig,
            state_dim: int,
            action_dim: int,
            device: str,
            goal_conditioned: bool,
            embed_dim: int,
            embed_pdrob: float,
            goal_seq_len: int,
            obs_seq_len: int,
            action_seq_len: int,
            goal_drop: float = 0.1,
            linear_output: bool = False,
    ):
        super().__init__()

        self.mamba = hydra.utils.instantiate(mamba)
        self.decoder = hydra.utils.instantiate(decoder)

        self.device = device
        self.goal_conditioned = goal_conditioned
        if not goal_conditioned:
            goal_seq_len = 0
        # input embedding stem
        # first we need to define the maximum block size
        # it consists of the goal sequence length plus 1 for the sigma embedding and 2 the obs seq len
        block_size = goal_seq_len + action_seq_len + obs_seq_len + 1
        # the seq_size is a little different since we have state action pairs for every timestep
        seq_size = goal_seq_len + obs_seq_len - 1 + action_seq_len

        self.tok_emb = nn.Linear(state_dim, embed_dim)
        self.tok_emb.to(self.device)

        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrob)
        self.drop.to(self.device)

        # needed for calssifier guidance learning
        self.cond_mask_prob = goal_drop

        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.embed_dim = embed_dim

        self.block_size = block_size
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len

        # we need another embedding for the time
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.time_emb.to(self.device)
        # get an action embedding
        self.action_emb = nn.Linear(action_dim, embed_dim)
        self.action_emb.to(self.device)
        # action pred module
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100),
                nn.GELU(),
                nn.Linear(100, self.action_dim)
            )
        # self.action_pred = nn.Linear(embed_dim, action_dim) # less parameters, worse reward
        self.action_pred.to(self.device)

        self.apply(self._init_weights)

        log.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
                len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
                len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    # x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, g: torch.Tensor
    # def forward(self, x, t, state, goal):
    def forward(
            self,
            actions,
            time,
            states,
            goals,
            uncond: Optional[bool] = False,
            keep_last_actions: Optional[bool] = False
    ):

        # actions = actions[:, self.obs_seq_len-1:, :]
        # states = states[:, :self.obs_seq_len, :]

        if len(states.size()) != 3:
            states = states.unsqueeze(0)

        if len(actions.size()) != 3:
            actions = actions.unsqueeze(0)

        b, t, dim = states.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # get the time embedding
        times = einops.rearrange(time, 'b -> b 1')
        emb_t = self.time_emb(times)

        if self.goal_conditioned:

            if self.training:
                goals = self.mask_cond(goals)
            # we want to use unconditional sampling during clasisfier free guidance
            if uncond:
                goals = torch.zeros_like(goals).to(self.device)

            goal_embed = self.tok_emb(goals)

        # embed them into linear representations for the transformer
        state_embed = self.tok_emb(states)
        action_embed = self.action_emb(actions)

        position_embeddings = self.pos_emb[:, :(t + self.goal_seq_len + self.action_seq_len - 1), :]
        # note, that the goal states are at the beginning of the sequence since they are available
        # for all states s_1, ..., s_t otherwise the masking would not make sense
        if self.goal_conditioned:
            goal_x = self.drop(goal_embed + position_embeddings[:, :self.goal_seq_len, :])

        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len + t), :])
        action_x = self.drop(action_embed + position_embeddings[:, self.goal_seq_len:(t + self.goal_seq_len + self.action_seq_len - 1), :])

        if t > 1:
            state = state_x[:, :(t-1), :]
            past_action = action_x[:, :(t-1), :]
            action_pred = action_x[:, (t-1):, :]

            sa_seq = torch.stack([state, past_action], dim=1).permute(0, 2, 1, 3).reshape(b, 2 * (t - 1),self.embed_dim)
            current_state = state_x[:, -1, :].unsqueeze(1)
            # emb_t, s_0, a_0, ..., s_(t-1), a_(t-1), s_t
            input_seq = torch.cat([emb_t, sa_seq, current_state], dim=1)
        else:
            state = state_x
            action_pred = action_x
            input_seq = torch.cat([emb_t, state], dim=1)

        # encode the state, goal and latent z into the hidden dim
        mamba_output = self.mamba(input_seq)
        m_b, m_t, _ = mamba_output.size()

        decoder_input = torch.cat([mamba_output, action_pred], dim=1)

        decoder_output = self.decoder(decoder_input)[:, m_t:, :]

        pred_actions = self.action_pred(decoder_output)

        return pred_actions

    def get_params(self):
        return self.parameters()


class Diffusion(nn.Module):

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            action_seq_len: int,
            model: DictConfig,
            beta_schedule: str,
            n_timesteps: int,
            loss_type: str,
            clip_denoised: bool,
            predict_epsilon=True,
            device: str = 'cuda',
            diffusion_x: bool = False,
            diffusion_x_M: int = 32,
    ) -> None:
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_seq_len = action_seq_len
        self.action_bounds = None
        # chose your beta style
        if beta_schedule == 'linear':
            self.betas = linear_beta_schedule(n_timesteps).to(self.device)
        elif beta_schedule == 'cosine':
            self.betas = cosine_beta_schedule(n_timesteps).to(self.device)
        elif beta_schedule == 'vp':
            # beta max: 10 beta min: 0.1
            self.betas = vp_beta_schedule(n_timesteps).to(self.device)

        self.model = hydra.utils.instantiate(model)
        self.n_timesteps = n_timesteps
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        # define alpha stuff
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), self.alphas_cumprod[:-1]])
        # required for forward diffusion q( x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod).to(self.device)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod).to(self.device)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod).to(self.device)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1).to(self.device)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod).to(
            self.device)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20)).to(self.device)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod).to(
            self.device)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
                    1. - self.alphas_cumprod).to(self.device)

        # either l1 or l2
        self.loss_fn = Losses[loss_type]()

        self.diffusion_x = diffusion_x
        self.diffusion_x_M = diffusion_x_M

    def get_params(self):
        '''
        Helper method to get all model parameters
        '''
        return self.model.get_params()

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        '''
        if self.predict_epsilon, model output is (scaled) noise, which is applied to compute the
        value for x_{t-1} otherwise the model can output x_{t-1} directly
        otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        """
        Computes the posterior mean and variance of the diffusion step at timestep t

        q( x_{t-1} | x_t, x_0)
        """
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, g: torch.Tensor, grad: bool = True):
        '''
        Predicts the denoised x_{t-1} sample given the current diffusion model

        :param x:  noisy input action
        :param t:  batch of timesteps
        :param s:  the current state observations batch
        :param grad:  bool, if the gradient should be computed

        :return:
            - the model mean prediction
            - the model log variance prediction
            - the posterior variance
        '''
        x_pred = x[:, -self.action_seq_len:, :]
        if grad:
            x_recon = self.predict_start_from_noise(x_pred, t=t, noise=self.model(x, t, s, g))
        else:
            x_recon = self.predict_start_from_noise(x_pred, t=t, noise=self.model_frozen(x, t, s, g))

        if self.clip_denoised:
            x_recon.clamp_(self.min_action, self.max_action)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x_pred, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, g: torch.Tensor, grad: bool = True):
        """
        Generated a denoised sample x_{t-1} given the trained model and noisy sample x_{t}

        :param x:  noisy input action
        :param t:  batch of timesteps
        :param s:  the current state observations batch
        :param grad:  bool, if the gradient should be computed

        :return:    torch.Tensor x_{t-1}
        """
        b, *_ = x.shape
        x_pred = x[:, -self.action_seq_len:, :]
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s, g=g, grad=grad)
        noise = torch.randn_like(x_pred)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_pred.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(self, state, actions, goal, shape, verbose=False, return_diffusion=False):
        """
        Main loop for generating samples using the learned model and the inverse diffusion step

        :param state: the current state observation
        :param shape: the shape of the action samples [B, D]
        :param return diffusion: bool, if the complete diffusion chain should be returned or not

        :return: either the predicted x_0 sample or a list with [x_{t-1}, .., x_{0}]
        """
        batch_size = shape[0]
        x = torch.randn(shape, device=self.device)

        if not torch.all(actions == 0):
            x = torch.cat([actions, x], dim=1)

        if return_diffusion:
            diffusion = [x]

        # start the inverse diffusion process to generate the action conditioned on the noise
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x_pre = self.p_sample(x, timesteps, state, goal)
            x[:, -self.action_seq_len:, :] = x_pre

            # if we want to return the complete chain add thee together
            if return_diffusion:
                diffusion.append(x)

        if self.diffusion_x:
            # The sampling process runs as normal for T denoising timesteps. The denoising timestep is then fixed,
            # t = 0, and extra denoising iterations continue to run for M timesteps. The intuition behind this
            # is that samples continue to be moved toward higher-likelihood regions for longer.
            # https://openreview.net/pdf?id=Pv1GPQzRrC8

            timesteps = torch.full((batch_size,), 0, device=self.device, dtype=torch.long)
            for m in range(0, self.diffusion_x_M):
                x = self.p_sample(x, timesteps, state, goal)
                if return_diffusion:
                    diffusion.append(x)

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x_pre

    def sample(self, state, actions, goal, *args, **kwargs):
        """
        Main Method to generate actions conditioned on the batch of state inputs

        :param state: the current state observation to conditon the diffusion model

        :return: x_{0} the predicted actions from the diffusion model
        """
        batch_size = state.shape[0]
        if len(state.shape) == 3:
            shape = (batch_size, self.model.action_seq_len, self.action_dim)
            # shape = (batch_size, state.shape[1], self.action_dim)
        else:
            shape = (batch_size, self.action_dim)
        action = self.p_sample_loop(state, actions, goal, shape, *args, **kwargs)
        return action.clamp_(self.min_action, self.max_action)

    def guided_p_sample(self, x, t, s, g, fun):
        '''
        Sample x_{t-1} from the model at the given timestep with additional conditioning
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param s: the current state observation
        :param fun: the conditoning model, subclass from torch.nn.Model

        :return: x_{t-1}
        '''
        b, *_ = x.shape
        with torch.no_grad():
            model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s, g=g)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        # Involve Function Guidance
        a = model_mean.clone().requires_grad_(True)
        q_value = fun(s, a)
        # q_value = q_value / q_value.abs().mean().detach()  # normalize q
        grads = torch.autograd.grad(outputs=q_value, inputs=a, create_graph=True, only_inputs=True)[0].detach()
        return (model_mean + model_log_variance * grads) + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def guided_sample(self, state: torch.Tensor, cond_fun, start: int = 0.2, verbose=False, return_diffusion=False):
        """
        Generated diffusion samples conditioned on cond_fun and starts deffusion at 20% of initial timesteps
        for faster generating process

        :param state: the current state batch
        :param cond_fun: the condition function to guide samples
        :param start:  defines the starting timestep 0 refers to starting sampling from t and 0.2 means
                       starting sampling from at 20% of the inital time step for faster sampling
        :return_diffusion: bool to decide, if only x_0 or all samples should be returned from the looop

        :return: either the predicted x_0 sample or a list with [x_{t-1}, .., x_{0}]
        """
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        x = torch.randn(shape, device=self.device)
        i_start = self.n_timesteps * start

        if return_diffusion: diffusion = [x]

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            if i <= i_start:
                x = self.guided_p_sample(x, timesteps, state, cond_fun)
            else:
                with torch.no_grad():
                    x = self.p_sample(x, timesteps, state)

            if return_diffusion: diffusion.append(x)

        x = x.clamp_(self.min_action, self.max_action)

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1).clamp_(self.min_action, self.max_action)
        else:
            return x

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise=None):
        """
        Main Method to sample the forward diffusion start with random noise and get
        the required values for the desired noisy sample at q(x_{t} | x_{0})
        at timestep t. The method is used for training only.

        :param x_start: the initial dataset batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.

        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start: torch.Tensor, state: torch.Tensor, goal: torch.Tensor, t: torch.Tensor, weights=1.0):
        """
        Computes the training loss of the diffusion model given a batch of data. At every
        training sample of the batch we generate noisy samples at different timesteps t
        and let the diffusion model predict the initial sample from the generated noisy one.
        The loss is computed by comparing the denoised sample from the diffusion model against
        the initial sample.

        :param x_start: the inital action samples
        :param state:   the current state observation batch
        :param t:       the batch of chosen timesteps
        :param weights: parameter to weight the individual losses

        :return loss: the total loss of the model given the input batch
        """
        x_pred = x_start[:, -self.action_seq_len:, :]
        x_past = x_start[:, :-self.action_seq_len, :]

        noise = torch.randn_like(x_pred)

        x_noisy = self.q_sample(x_start=x_pred, t=t, noise=noise)

        x = torch.cat([x_past, x_noisy], dim=1)
        x_recon = self.model(x, t, state, goal)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss

    def loss(self, x: torch.Tensor, state: torch.Tensor, goal: Optional[torch.Tensor] = None, weights=1.0):
        '''
        Computes the batch loss for the diffusion model
        '''
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, state, goal, t, weights)

    def forward(self, state, actions, goal, *args, **kwargs):
        '''
        General forward method, which generates samples given the chosen input state
        '''
        return self.sample(state, actions, goal, *args, **kwargs)

    def sample_t_middle(self, state: torch.Tensor, goal):
        """
        Fast generation of new samples, which only use 20% of the denoising steps of the true
        denoising process
        """
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)

        batch_size = shape[0]
        x = torch.randn(shape, device=self.device)

        t = np.random.randint(0, int(self.n_timesteps * 0.2))
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state, goal, grad=(i == t))
        action = x
        return action.clamp_(self.min_action, self.max_action)

    def sample_t_last(self, state: torch.Tensor, goal):
        """
        Generate denoised samples with all denoising steps
        """
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)

        x = torch.randn(shape, device=self.device)
        cur_T = np.random.randint(int(self.n_timesteps * 0.8), self.n_timesteps)
        for i in reversed(range(0, cur_T)):
            timesteps = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            if i != 0:
                with torch.no_grad():
                    x = self.p_sample(x, timesteps, state, goal)
            else:
                x = self.p_sample(x, timesteps, state, goal)

        action = x
        return action.clamp_(self.min_action, self.max_action)

    def sample_last_few(self, state: torch.Tensor, goal):
        """
        Return samples, that have not complelty denoised the data. The inverse diffusion process
        is stopped 5 timetsteps before the true denoising.
        """
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)

        x = torch.randn(shape, device=self.device)
        nest_limit = 5
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            if i >= nest_limit:
                with torch.no_grad():
                    x = self.p_sample(x, timesteps, state, goal)
            else:
                x = self.p_sample(x, timesteps, state, goal)

        action = x
        return action.clamp_(self.min_action, self.max_action)



class DiffusionPolicy(nn.Module):
    def __init__(self, model: DictConfig, obs_encoder: DictConfig, visual_input: bool = True, device: str = "cpu"):
        super(DiffusionPolicy, self).__init__()

        self.visual_input = visual_input

        self.obs_encoder = hydra.utils.instantiate(obs_encoder).to(device)

        self.model = hydra.utils.instantiate(model).to(device)

    def forward(self, inputs, goal, action=None, if_train=False, if_return_obs=False):
        # encode state and visual inputs
        # the encoder should be shared by all the baselines

        if self.visual_input:
            agentview_image, in_hand_image, past_actions = inputs

            B, T, C, H, W = agentview_image.size()

            agentview_image = agentview_image.view(B * T, C, H, W)
            in_hand_image = in_hand_image.view(B * T, C, H, W)
            # state = state.view(B * T, -1)

            # bp_imgs = einops.rearrange(bp_imgs, "B T C H W -> (B T) C H W")
            # inhand_imgs = einops.rearrange(inhand_imgs, "B T C H W -> (B T) C H W")

            obs_dict = {"agentview_rgb": agentview_image,
                        "eye_in_hand_rgb": in_hand_image,}
                        # "robot_ee_pos": state}

            obs = self.obs_encoder(obs_dict)
            obs = obs.view(B, T, -1)

        if if_train:
            action = torch.cat([past_actions, action], dim=1)
            return self.model.loss(action, obs, goal)

        # make prediction
        pred = self.model(obs, past_actions, goal)

        if if_return_obs:
            return pred, obs

        return pred

    def get_params(self):
        return self.parameters()

class DiffusionAgent(BaseAgent):

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

        # lora.mark_only_lora_as_trainable(self.model)

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

        self.diffusion_kde = diffusion_kde
        self.diffusion_kde_samples = diffusion_kde_samples

        self.obs_seq_len = obs_seq_len
        self.action_seq_size = action_seq_size
        self.action_counter = self.action_seq_size

        self.bp_image_context = deque(maxlen=self.obs_seq_len)
        self.inhand_image_context = deque(maxlen=self.obs_seq_len)
        self.des_robot_pos_context = deque(maxlen=self.window_size)

        self.obs_context = deque(maxlen=self.obs_seq_len)
        self.goal_context = deque(maxlen=self.goal_window_size)

        # if we use DiffusionGPT we need an action context
        if not self.pred_last_action_only:
            self.action_context = deque(maxlen=self.obs_seq_len - 1)
            self.que_actions = True
        else:
            self.que_actions = False

    def train_agent(self):

        best_test_mse = 1e10

        for num_epoch in tqdm(range(self.epoch)):

            # run a test batch every n steps
            if not (num_epoch+1) % self.eval_every_n_epochs:

                test_mse = []
                for data in self.test_dataloader:

                    if self.goal_condition:
                        state, action, mask, goal = data
                        mean_mse = self.evaluate(state, action, goal)
                    else:
                        state, action, mask = data
                        mean_mse = self.evaluate(state, action)

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

                if self.goal_condition:
                    state, action, mask, goal = data
                    batch_loss = self.train_step(state, action, goal)
                else:
                    state, action, mask = data
                    batch_loss = self.train_step(state, action)

                train_loss.append(batch_loss)

                wandb.log(
                    {
                        "loss": batch_loss,
                    }
                )

        self.store_model_weights(self.working_dir, sv_name=self.last_model_name)

        log.info("Training done!")

    def train_vision_agent(self):

        for num_epoch in tqdm(range(self.epoch)):

            train_loss = []

            for data in self.train_dataloader:
                bp_imgs, inhand_imgs, action, mask = data

                bp_imgs = bp_imgs.to(self.device)
                inhand_imgs = inhand_imgs.to(self.device)

                # obs = self.scaler.scale_input(obs)
                action = self.scaler.scale_output(action)

                past_action = action[:, :(self.obs_seq_len-1), :].contiguous()
                pred_action = action[:, (self.obs_seq_len-1):, :].contiguous()

                bp_imgs = bp_imgs[:, :self.obs_seq_len].contiguous()
                inhand_imgs = inhand_imgs[:, :self.obs_seq_len].contiguous()

                state = (bp_imgs, inhand_imgs, past_action)

                batch_loss = self.train_step(state, pred_action)

                train_loss.append(batch_loss)

                wandb.log({"train_loss": batch_loss.item()})

            log.info("Epoch {}: Mean train loss is {}".format(num_epoch, batch_loss.item()))

        log.info("training done")
        self.store_model_weights(self.working_dir, sv_name=self.last_model_name)

    def train_step(self, state: torch.Tensor, action: torch.Tensor, goal: Optional[torch.Tensor] = None) -> float:

        # state = state.to(self.device).to(torch.float32)  # [B, V]
        # action = action.to(self.device).to(torch.float32)  # [B, D]
        # scale data if necessarry, otherwise the scaler will return unchanged values
        self.model.train()

        # state = self.scaler.scale_input(state)
        # action = self.scaler.scale_output(action)

        if goal is not None:
            goal = self.scaler.scale_input(goal)

        # action = action[:, self.obs_seq_len-1:, :]
        # state = state[:, :self.obs_seq_len, :]

        # Compute the loss.
        loss = self.model(state, goal, action=action, if_train=True)
        # Before the backward pass, zero all the network gradients
        self.optimizer.zero_grad()
        # Backward pass: compute gradient of the loss with respect to parameters
        loss.backward()
        # Calling the step function to update the parameters
        self.optimizer.step()

        self.steps += 1

        # update the ema model
        if self.steps % self.update_ema_every_n_steps == 0:
            self.ema_helper.update(self.model.parameters())
        return loss

    @torch.no_grad()
    def evaluate(
            self, state: torch.tensor, action: torch.tensor, goal: Optional[torch.Tensor] = None
    ) -> float:

        # scale data if necessarry, otherwise the scaler will return unchanged values
        state = self.scaler.scale_input(state)
        action = self.scaler.scale_output(action)

        action = action[:, self.obs_seq_len - 1:, :]
        state = state[:, :self.obs_seq_len, :]

        if goal is not None:
            goal = self.scaler.scale_input(goal)

        total_mse = 0.0
        # use the EMA model variant
        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())

        self.model.eval()

        # Compute the loss.
        loss = self.model.loss(action, state, goal)

        # model_pred = self.model(state, goal)
        # mse = nn.functional.mse_loss(model_pred, action, reduction="none")

        total_mse += loss.mean().item()

        # restore the previous model parameters
        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())
        return total_mse

    def reset(self):
        """ Resets the context of the model."""
        self.obs_context.clear()
        self.action_counter = self.action_seq_size

        self.bp_image_context.clear()
        self.inhand_image_context.clear()
        self.des_robot_pos_context.clear()
        self.action_context.clear()

    @torch.no_grad()
    def predict(self, state: torch.Tensor, goal: Optional[torch.Tensor] = None, extra_args=None, if_vision=True) -> torch.Tensor:
        # scale data if necessarry, otherwise the scaler will return unchanged values

        if if_vision:
            bp_image, inhand_image = state

            bp_image = torch.from_numpy(bp_image).to(self.device).float().permute(2, 0, 1).unsqueeze(0) / 255.
            inhand_image = torch.from_numpy(inhand_image).to(self.device).float().permute(2, 0, 1).unsqueeze(0) / 255.

            self.bp_image_context.append(bp_image)
            self.inhand_image_context.append(inhand_image)

            if len(self.action_context) == 0:
                past_actions = torch.zeros(1, 1, 7).to(self.device)
            else:
                past_actions = torch.stack(tuple(self.action_context), dim=1)

            bp_image_seq = torch.stack(tuple(self.bp_image_context), dim=1)
            inhand_image_seq = torch.stack(tuple(self.inhand_image_context), dim=1)
            # des_robot_pos_seq = torch.stack(tuple(self.des_robot_pos_context), dim=1)

            input_state = (bp_image_seq, inhand_image_seq, past_actions)
        else:
            obs = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
            obs = self.scaler.scale_input(obs)
            self.obs_context.append(obs)
            input_state = torch.stack(tuple(self.obs_context), dim=1)  # type: ignore

        if self.action_counter == self.action_seq_size:
            self.action_counter = 0

            if self.use_ema:
                self.ema_helper.store(self.model.parameters())
                self.ema_helper.copy_to(self.model.parameters())

            self.model.eval()

            model_pred = self.model(input_state, goal)

            # restore the previous model parameters
            if self.use_ema:
                self.ema_helper.restore(self.model.parameters())
            inv_model_pred = self.scaler.inverse_scale_output(model_pred)

            self.model_pred_action = model_pred
            self.curr_action_seq = inv_model_pred

        next_action = self.curr_action_seq[:, self.action_counter, :]

        self.action_context.append(self.model_pred_action[:, self.action_counter, :])

        self.action_counter += 1
        return next_action.detach().cpu().numpy()

    def random_sample(self, state: torch.Tensor, goal: Optional[torch.Tensor] = None, extra_args=None, if_vision=True):
        # scale data if necessarry, otherwise the scaler will return unchanged values

        if if_vision:
            bp_imgs, inhand_imgs = state

            bp_imgs = bp_imgs[:, :self.obs_seq_len].contiguous()
            inhand_imgs = inhand_imgs[:, :self.obs_seq_len].contiguous()

            input_state = (bp_imgs, inhand_imgs)
        else:
            obs = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
            obs = self.scaler.scale_input(obs)
            self.obs_context.append(obs)
            input_state = torch.stack(tuple(self.obs_context), dim=1)  # type: ignore

        # self.model.eval()
        # do default model evaluation
        model_pred, obs_embedding = self.model(input_state, goal, if_return_obs=True)

        return model_pred, obs_embedding
    @torch.no_grad()
    def load_pretrained_model(self, weights_path: str, sv_name=None, **kwargs) -> None:
        """
        Method to load a pretrained model weights inside self.model
        """
        # self.model.load_state_dict(torch.load(os.path.join(weights_path, "model_state_dict.pth")))
        self.model.load_state_dict(torch.load(os.path.join(weights_path, sv_name)), strict=False)
        self.ema_helper = ExponentialMovingAverage(self.model.parameters(), self.decay, self.device)
        log.info('Loaded pre-trained model parameters')

    @torch.no_grad()
    def store_model_weights(self, store_path: str, sv_name=None) -> None:
        """
        Store the model weights inside the store path as model_weights.pth
        """
        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())
        # torch.save(self.model.state_dict(), os.path.join(store_path, "model_state_dict.pth"))
        torch.save(self.model.state_dict(), os.path.join(store_path, sv_name))
        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())
        torch.save(self.model.state_dict(), os.path.join(store_path, "non_ema_model_state_dict.pth"))