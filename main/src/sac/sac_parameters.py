"""Container for holding training hyperparameters"""

import mlflow
import os
import json

import numpy as np


class Parameters:
    """Holds parameters for sac training"""

    params_directory = "../../../params/sac"

    def __init__(self, inputs: int, outputs: int, fname: str):
        """
        Initializes the parameters with default values.
        These values should NOT be changed in this class but on individual instances of it.
        :param inputs: The size on the input layer of the actor critic network.
        :param outputs: The size of the output layer of actions.
        """
        # name of the run which is used to create the folder where the trained model is saved
        self.file_name = fname

        # actor critic parameters
        # the size of the state inputs
        self.inputs = inputs

        # the number of output actions
        self.outputs = outputs

        # Number of hidden units per layer
        self.hidden_dim = 256

        # Learning rates for each network
        self.value_lr = 3e-4
        self.soft_q_lr = 3e-4
        self.policy_lr = 3e-4

        # update parameters
        # Size of the replay buffer
        self.replay_buffer_size = 500000

        # Size of the batch that is used for the update
        self.batch_size = 256

        # the soft update coefficient (“polyak update”, between 0 and 1) for the target network
        self.soft_tau = 5e-3

        # the discount factor applied to the rewards in each timestep
        self.gamma = 0.99

        # training parameters
        # training episodes
        self.max_episodes = 150000

        # Maximum steps per episode
        self.max_steps = 5000000

    def log_to_mlflow(self):
        """
        Saves the parameters to the mlflow server
        """
        mlflow.log_param('training iterations', self.max_episodes)
        mlflow.log_param('batch size', self.batch_size)
        mlflow.log_param('max steps', self.max_steps)
        mlflow.log_param('gamma', self.gamma)
        mlflow.log_param('soft tau', self.soft_tau)
        mlflow.log_param('replay_buffer_size', self.replay_buffer_size)
        mlflow.log_param('value learning rate', self.value_lr)
        mlflow.log_param('soft q learning rate', self.soft_q_lr)
        mlflow.log_param('policy learning rate', self.policy_lr)

    def load(self, file_name: str):
        dirname = os.path.dirname(__file__)
        with open(os.path.join(dirname, Parameters.params_directory, file_name)) as file:
            params_dict = json.loads(file.read())

        for key in self.__dict__:
            if key in params_dict:
                self.__dict__[key] = params_dict[key]
            key_range = f"{key}_range"

            if key_range in params_dict:
                self.__dict__[key] = np.random.uniform(*params_dict[key_range])

        self.max_episodes = int(self.max_episodes)
