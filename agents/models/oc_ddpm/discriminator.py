import math

import hydra
import torch
from einops import einops
from omegaconf import DictConfig
from torch import nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, action_encoder: DictConfig, decoder: DictConfig,
                 action_dim, seq_size, embed_dim, device: str,):
        super(Discriminator, self).__init__()

        self.device = device

        self.action_emb = nn.Linear(action_dim, embed_dim)
        # self-att to get action embedding
        self.action_encoder = hydra.utils.instantiate(action_encoder)

        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size+1, embed_dim))

        self.cls_emb = nn.Parameter(torch.zeros(1, embed_dim))

        # self.time_emb = nn.Sequential(
        #     SinusoidalPosEmb(embed_dim),
        #     nn.Linear(embed_dim, embed_dim * 2),
        #     nn.Mish(),
        #     nn.Linear(embed_dim * 2, embed_dim),
        # )
        # self.time_emb.to(self.device)

        # cross-att between action and obs
        self.decoder = hydra.utils.instantiate(decoder)

        # predict the score
        self.score_pred = nn.Sequential(nn.Linear(embed_dim, 1),
                                        nn.Sigmoid())

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, obs, action):

        bs = action.shape[0]

        cls_emb = torch.unsqueeze(self.cls_emb, axis=0).repeat(bs, 1, 1)

        act_emb = self.action_emb(action)
        # act_emb = torch.cat([cls_emb, act_emb], dim=1)
        act_emb = act_emb + self.pos_emb[:, :act_emb.shape[1]]
        act_emb = self.action_encoder(act_emb)

        act_emb = torch.cat([act_emb, cls_emb], dim=1)

        out = self.decoder(act_emb, obs)

        score = self.score_pred(out[:, -1, :])

        return score

    # def forward(self, obs, action):
    #
    #     bs = action.shape[0]
    #
    #     cls_emb = torch.unsqueeze(self.cls_emb, axis=0).repeat(bs, 1, 1)
    #
    #     act_emb = self.action_emb(action)
    #     act_emb = torch.cat([cls_emb, act_emb], dim=1)
    #     act_emb = act_emb + self.pos_emb[:, :act_emb.shape[1]]
    #
    #     act_emb = self.action_encoder(act_emb)
    #
    #     out = self.decoder(act_emb, obs)
    #
    #     score = self.score_pred(out[:, 0, :])
    #
    #     return score


# Generator Model
class Generator(nn.Module):
    def __init__(self, encoder: DictConfig):
        super(Generator, self).__init__()
        self.encoder = hydra.utils.instantiate(encoder)

    def forward(self, z):
        output = self.model(z)
        output = output.view(output.size(0), 1, 28, 28)
        return output