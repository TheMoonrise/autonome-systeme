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

    def model_path(self, appendix: str = '', is_save: bool = True):
        """
        Provides the path at which the model will be saved.
        :param appendix: An appendix to add to the file name of the model.
        :param is_save: Whether the save or load path should be provided.
        """
        directory = os.path.dirname(__file__)
        path = ActorCritic.model_directory_save if is_save else ActorCritic.model_directory_load
        return os.path.join(directory, path, self.name + appendix)

    def save(self, appendix: str = ''):
        """
        Saves the current model parameters to file.
        :param appendix: An appendix to add to the file name of the model.
        """
        path = self.model_path(appendix, is_save=True)
        torch.save(self.state_dict(), path)

    def load(self, device: str = 'cpu'):
        """
        Loads model parameters from the name of the model.
        :param device: The device on which to run the model
        """
        path = self.model_path(is_save=False)
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval()
