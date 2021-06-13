"""Implementation of an actor critic network for ppo"""

import torch
import os

from torch import nn
from .ppo_parameters import Parameters


class ActorCritic(nn.Module):
    """Actor critic module to learn actions as well as state values"""

    model_directory_save = "../../../models/ppo/temp"
    model_directory_load = "../../../models/ppo"

    def __init__(self, params: Parameters, name: str):
        """
        Initializes the model.
        :param params: Parameters container holding setup parameters.
        :param name: The name of the model used when saving.
        """
        super().__init__()

        self.name = name
        self.params = params

    def save(self, appendix: str = ''):
        """
        Saves the current model parameters to file.
        :param appendix: An appendix to add to the file name of the model.
        """
        directory = os.path.dirname(__file__)
        path = os.path.join(directory, ActorCritic.model_directory_save, self.name + appendix)
        torch.save(self.state_dict(), path)

    def load(self, device: str ='cpu'):
        """
        Loads model parameters from the name of the model.
        :param device: The device on which to run the model
        """
        directory = os.path.dirname(__file__)
        path = os.path.join(directory, ActorCritic.model_directory_load, self.name)
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval()