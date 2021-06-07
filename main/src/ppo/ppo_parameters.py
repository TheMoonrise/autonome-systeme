"""Container for holding training hyperparameters"""


class Parameters:
    """Holds parameters for ppo training"""

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

        # the size of the hidden layers
        self.hidden01 = 64
        self.hidden02 = 256

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
        self.tau = 1

        # training parameters
        # the number of iterations the training should run for
        self.training_iterations = 2500

        # the trace length to collect for each iteration
        self.trace = 35

        # the learning rate of the optimizer
        self.learning_rate = 1e-5
