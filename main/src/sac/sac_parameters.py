"""Container for holding training hyperparameters"""

import mlflow


class Parameters:
    """Holds parameters for sac training"""
    def __init__(self, inputs: int, outputs: int, episodes: int, name: str):
        """
        Initializes the parameters with default values.
        These values should NOT be changed in this class but on individual instances of it.
        :param inputs: The size on the input layer of the actor critic network.
        :param outputs: The size of the output layer of actions.
        """
        # name of the run which is used to create the folder where the trained model is saved
        self.name = name

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
        self.max_episodes = episodes

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
