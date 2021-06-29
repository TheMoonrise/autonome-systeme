"""Implementation of an actor critic network for ppo"""

import torch
import os
import random

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
        self.id = f'i{random.randint(1, 9999):0>4}'

    def model_directory(self, is_save: bool):
        """
        Provides the path to the model directory.
        :param is_save: Whether the save or load path should be provided.
        :return: The path to the model directory.
        """
        directory = os.path.dirname(__file__)

        if is_save: path = os.path.join(ActorCritic.model_directory_save, self.id)
        else: path = ActorCritic.model_directory_load
        path = os.path.join(directory, path)

        os.makedirs(path, exist_ok=True)
        return path

    def model_path(self, appendix: str = '', is_save: bool = True):
        """
        Provides the path at which the model will be saved.
        :param appendix: An appendix to add to the file name of the model.
        :param is_save: Whether the save or load path should be provided.
        :return: The model path including the model file name.
        """
        directory = self.model_directory(is_save)
        if is_save: appendix = f'-{self.id}{appendix}'
        return os.path.join(directory, self.name + appendix)

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
