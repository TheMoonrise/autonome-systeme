import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        """
        Initializes a Value Network.
        :param state_dim: Dimension of the input layer (size of the observation space)
        :param hidden_dim: Dimension for the networks hidden layer
        :param init_w: Default weights for the network
        """
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        """
        Forward propagation of the state through the value network
        :param state: input state
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        """
        Initializes the Soft Q Network.
        :param num_inputs: size of the observation space
        :param num_actions: size of the action space
        :param hidden_size: Size for the networks hidden layers
        :param init_w: initial values for the weights of the network
        """
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        """
        Forward propagation of the state through the q network
        :param state: input state
        """
        # concatenates state and action in the dimension 2
        # tensor indexes start at 0 in PyTorch, so this refers to the 3rd dimension because state and action are 3-dimensional
        x = torch.cat([state, action], 2)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x  # returns tensor of shape 128x10x1 because linear 3 has output 1


class PolicyNetwork(nn.Module):
    model_directory_save = "../../../models/sac/temp/"

    def __init__(self, num_inputs, num_actions, hidden_size, name: str, device, params,
                 init_w=3e-3, log_std_min=-20, log_std_max=2):
        """
        Initializes the Policy Network.
        :param num_inputs: size of the observation space
        :param num_actions: size of the action space
        :param hidden_size: Size for the networks hidden layers
        :param init_w: initial values for the weights of the network
        :param log_std_min: min value for the log standard deviation so that it is in a reasonable size
        :param log_std_max: max value for the log standard deviation so that it is in a reasonable size
        """
        super(PolicyNetwork, self).__init__()

        self.device = device

        self.name = name

        self.model_folder = params.file_name + "/"
        directory = os.path.dirname(__file__)
        path = os.path.join(directory, PolicyNetwork.model_directory_save, self.model_folder)
        try:
            os.mkdir(path)
        except FileExistsError:
            print("folder already exists, old data might be overwritten")

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        """
        Forward propagation of the state through the policy network
        :param state: input state
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        # mean and log_std have torch.Size([1, 10, 20])
        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        """
        evaluates the policy based on the current state
        :param state: current state
        :param epsilon:
        :returns: set of parameters for the policy network update
        """
        # Calculate Gaussian distribution of (mean, log_std)
        # state has torch.Size([128, 10, 158])
        # mean, log_std torch.Size([128, 10, 20]) because policy_net has 20 num_action
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # a~′θ(s,ξ)=tanh(μθ(s)+σθ(s)⊙ξ)
        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(mean + std * z.to(self.device))

        # Calculate entropies
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(self.device)) - torch.log(1 - action.pow(2) + epsilon)

        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        """
        Returns the action based on a squashed gaussian policy.
        That means the samples are obtained according to:
            a(s,e)= tanh(mu(s)+sigma(s)+e)
        :param state: current state for which an action should be chosen
        :returns: returns the chosen action
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # calculate Gaussian distribusion of (mean, log_std)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        # sample actions
        z = normal.sample().to(self.device)
        action = torch.tanh(mean + std * z)

        action = action.cpu()  # .detach().cpu().numpy()
        return action[0]

    def save(self, appendix: str = ''):
        """
        Saves the current model parameters to file.
        :param appendix: An appendix to add to the file name of the model
        """
        directory = os.path.dirname(__file__)
        path = os.path.join(directory, PolicyNetwork.model_directory_save, self.model_folder, self.name + appendix)
        torch.save(self.state_dict(), path)
        return path
