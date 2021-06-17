"""Container for holding training hyperparameters"""

import json
import os
import numpy as np


class Parameters:
    """Holds parameters for ppo training"""

    params_directory = "../../../params/ppo"

    def __init__(self, inputs: int, outputs: int):
        """
        Initializes the parameters with default values.
        These values should NOT be changed in this class but on individual instances of it.
        :param inputs: The size on the input layer of the actor critic network.
        :param outputs: The size of the output layer of actions.
        """

        # actor critic parameters
        # the size of the state inputs
        self.inputs = inputs

        # the number of output actions
        self.outputs = outputs

        # update parameters
        # the range to which the ratio between old and new policy is clamped
        self.clip = 0.2

        # the number of optimizations epochs to perform on one batch of traces
        self.epochs = 10

        # the size of the minibatches in which the update batch is divided into
        self.mini_batch_size = 25

        # the factor at which the critic loss influences the overall loss
        self.influence_critic = 0.5

        # the factor at which the entropy influences the overall loss
        self.influence_entropy = 0.001

        # return value parameters
        # the discount factor applied to the rewards in each timestep
        self.gamma = 0.9

        # the discount factor applied additionally to gamma but only on the final sum elements
        self.lmbda = 1

        # training parameters
        # the number of iterations the training should run for
        self.training_iterations = 50_000

        # the trace length to collect for each iteration
        self.trace = 35

        # the learning rate of the optimizer
        self.learning_rate = 1e-5

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
