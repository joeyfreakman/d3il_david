import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import abc

from typing import Optional, Union

import agents.models.bet.utils as utils


class AbstractActionAE(utils.SaveModule, abc.ABC):
    @abc.abstractmethod
    def fit_model(
        self,
        input_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        obs_encoding_net: Optional[nn.Module] = None,
    ) -> None:
        pass

    @abc.abstractmethod
    def encode_into_latent(
        self,
        input_action: torch.Tensor,
        input_rep: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Given the input action, discretize it.

        Inputs:
        input_action (shape: ... x action_dim): The input action to discretize. This can be in a batch,
        and is generally assumed that the last dimnesion is the action dimension.

        Outputs:
        discretized_action (shape: ... x num_tokens): The discretized action.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def decode_actions(
        self,
        latent_action_batch: Optional[torch.Tensor],
        input_rep_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Given a discretized action, convert it to a continuous action.

        Inputs:
        latent_action_batch (shape: ... x num_tokens): The discretized action
        generated by the discretizer.

        Outputs:
        continuous_action (shape: ... x action_dim): The continuous action.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_latents(self) -> Union[int, float]:
        """
        Number of possible latents for this generator, useful for state priors that use softmax.
        """
        return float("inf")