"""Container for holding training hyperparameters"""


class Parameters:
    """Holds parameters for sac training"""
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

        # Number of hidden units per layer
        self.hidden_dim = 256

        # Learning rates for each network
        self.value_lr = 3e-4
        self.soft_q_lr = 3e-4
        self.policy_lr = 3e-4

        # update parameters
        # Size of the replay buffer
        self.replay_buffer_size = 5000000

        # Size of the batch that is used for the update
        self.batch_size = 256

        # the soft update coefficient (“polyak update”, between 0 and 1) for the target network
        self.soft_tau = 5e-3

        # the discount factor applied to the rewards in each timestep
        self.gamma = 0.99

        # training parameters
        # training episodes
        self.max_episodes = 4000

        # Maximum steps per episode
        self.max_steps = 5000000
