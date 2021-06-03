"""Implementation of an actor critic network for ppo"""

import torch
from torch import nn
from torch.distributions import Normal
from typing import Union


class ActorCriticPPO(nn.Module):
    """Actor critic module to learn actions as well as state values"""

    def __init__(self, num_inputs: int, num_outputs: int, num_hidden: int, scale: float = 1):
        super().__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_outputs)
        )

        # maybe we must add an empty dimension at the start here
        self.scale = nn.Parameter(torch.ones(num_outputs) * scale)

        self.apply(self._configure_weights)

    def _configure_weights(self, x):
        """
        Configures the initial weights for the given module.
        :param x: The module to set the weights for.
        Only Linear modules are considered.
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
        scl = self.scale.expand_as(loc)
        dst = Normal(loc, scl)
        return dst, val
