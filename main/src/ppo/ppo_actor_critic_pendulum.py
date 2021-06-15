"""Implementation of an actor critic network for ppo"""

import torch

from torch import nn
from torch.distributions import Normal
from typing import Union
from .ppo_parameters import Parameters
from .ppo_actor_critic import ActorCritic


class ActorCriticPendulum(ActorCritic):
    """Actor critic module to learn actions as well as state values"""

    def __init__(self, params: Parameters, name: str):
        """
        Initializes the model.
        :param params: Parameters container holding setup parameters.
        :param name: The name of the model used when saving.
        """
        super().__init__(params, name)

        self.hidden01 = 64
        self.hidden02 = 256

        self.actor = nn.Sequential(
            nn.Linear(params.inputs, self.hidden01),
            nn.ReLU(),
            nn.Linear(self.hidden01, self.hidden02),
            nn.ReLU()
        )

        self.actor_head_loc = nn.Sequential(
            nn.Linear(self.hidden02, params.outputs),
            nn.Tanh()
        )

        self.actor_head_scl = nn.Sequential(
            nn.Linear(self.hidden02, params.outputs),
            nn.Softplus()
        )

        self.critic = nn.Sequential(
            nn.Linear(params.inputs, self.hidden01),
            nn.ReLU(),
            nn.Linear(self.hidden01, self.hidden02),
            nn.ReLU(),
            nn.Linear(self.hidden02, 1)
        )

    def forward(self, x: torch.Tensor) -> Union[torch.distributions.Normal, torch.Tensor]:
        """
        Computes the action distribution and state value for the given batch of states.
        :param x: The batch of states.
        :returns: The normal distribution over the possible action values.
        :returns: The state value obtained from the critic network.
        """
        act = self.actor(x)
        # the action amplitude for the pendulum is 4
        loc = self.actor_head_loc(act) * 2
        scl = self.actor_head_scl(act) + 1e-3
        dst = Normal(loc, scl)

        val = self.critic(x)
        return dst, val
