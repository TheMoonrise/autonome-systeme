"""Implementation of an actor critic network for ppo"""

import torch
import os

from torch import nn
from torch.distributions import Normal
from typing import Union
from .ppo_parameters import Parameters


class ActorCritic(nn.Module):
    """Actor critic module to learn actions as well as state values"""

    model_directory_save = "../../../models/ppo/temp"
    model_directory_load = "../../../models/ppo"

    def __init__(self, params: Parameters):
        """
        Initializes the model.
        :param params: Parameters container holding setup parameters.
        """
        super().__init__()

        self.net = nn.Sequential(
            # nn.Linear(params.inputs, params.hidden01),
            # nn.ReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(params.inputs, params.hidden01),
            nn.ReLU(),
            nn.Linear(params.hidden01, params.hidden02),
            nn.ReLU()
        )

        self.actor_head_loc = nn.Sequential(
            nn.Linear(params.hidden02, params.outputs),
            # check if this is sensible for the crawler domain
            nn.Tanh()
        )

        self.actor_head_scl = nn.Sequential(
            nn.Linear(params.hidden02, params.outputs),
            nn.Softplus()
        )

        self.critic = nn.Sequential(
            nn.Linear(params.inputs, params.hidden01),
            nn.ReLU(),
            nn.Linear(params.hidden01, params.hidden02),
            nn.ReLU(),
            nn.Linear(params.hidden02, 1)
        )

        # self.apply(self._configure_weights)

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
        x = self.net(x)

        act = self.actor(x)
        # the action amplitude for the pendulum is 4
        loc = self.actor_head_loc(act) * 2
        scl = self.actor_head_scl(act) + 1e-3
        dst = Normal(loc, scl)

        val = self.critic(x)
        return dst, val

    def save(self, file_name: str):
        """
        Saves the current model parameters to file.
        :param file_name: The name of the file the model is saved to.
        """
        directory = os.path.dirname(__file__)
        path = os.path.join(directory, ActorCritic.model_directory_save, file_name)
        torch.save(self.state_dict(), path)

    def load(self, file_name: str):
        """
        Loads model parameters from the given file name.
        : param file_name: the name of the file the model is loaded from.
        """
        directory = os.path.dirname(__file__)
        path = os.path.join(directory, ActorCritic.model_directory_load, file_name)
        self.load_state_dict(torch.load(path))
        self.eval()
