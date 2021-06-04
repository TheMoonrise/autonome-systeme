"""Implementation of an actor critic network for ppo"""

import torch
from torch import nn
from torch.distributions import Normal
from typing import Union
from .ppo_parameters import Parameters


class ActorCritic(nn.Module):
    """Actor critic module to learn actions as well as state values"""

    def __init__(self, params: Parameters):
        """
        Initializes the model.
        :param params: Parameters container holding setup parameters.
        """
        super().__init__()

        self.critic = nn.Sequential(
            nn.Linear(params.inputs, params.hidden),
            nn.ReLU(),
            nn.Linear(params.hidden, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(params.inputs, params.hidden),
            nn.ReLU(),
            nn.Linear(params.hidden, params.outputs)
        )

        # maybe we must add an empty dimension at the start here
        self.scale = nn.Parameter(torch.ones(params.outputs) * params.scale)

        self.apply(self._configure_weights)

    def _configure_weights(self, x):
        """
        Configures the initial weights for the given module.
        :param x: The module to set the weights for.
        Only linear modules are considered.
        """
        if isinstance(x, nn.Linear):
            nn.init.normal_(x.weight, mean=0, std=0.1)
            nn.init.constant_(x.bias, 0.1)

    def forward(self, x: torch.Tensor) -> Union[torch.distributions.Normal, torch.Tensor]:
        """
        Computes the action distribution and state value for the given batch of states.
        :param x: The batch of states.
        :returns: The normal distribution over the possible action values.
        :returns: The state value obtained from the critic network.
        """
        val = self.critic(x)
        loc = self.actor(x)
        scl = self.scale.exp().expand_as(loc)
        dst = Normal(loc, scl)
        return dst, val
