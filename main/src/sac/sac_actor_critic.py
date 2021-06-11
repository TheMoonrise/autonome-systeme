import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ValueNetwork(nn.Module):
    """
    Initializes the model.
    :param state_dim: Dimension of each state.
    :param hidden_dim: Dimension for the Networks hidden layers.
    :param init_w: Default weight.
    """
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
    
    """
    TODO: Add comment
    """
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        
        
class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        """
        Initializes the model.
        :param num_inputs:
        :param num_actions:
        :param hidden_size: Size for the Networks hidden layers.
        :param init_w: Default weight.
        """
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    """
    TODO: Add comment
    """    
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        
        
class PolicyNetwork(nn.Module):
    model_directory_save = "../../../models/sac/temp"
    """
    Initializes the model.
    :param num_inputs:
    :param num_actions:
    :param hidden_size: Size for the Networks hidden layers.
    :param init_w:
    :param log_std_min:
    :param log_std_max:
    """
    def __init__(self, num_inputs, num_actions, hidden_size, name: str, device, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.device = device

        self.name = name
        
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

    """
    """
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    """
    """
    def evaluate(self, state, epsilon=1e-6):
        # calculate Gaussian distribusion of (mean, log_std)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample()
        action = torch.tanh(mean+ std*z.to(self.device))
        # calculate entropies
        log_prob = Normal(mean, std).log_prob(mean+ std*z.to(self.device)) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob, z, mean, log_std
        
    """
    Returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
    """
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # calculate Gaussian distribusion of (mean, log_std)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        # sample actions
        z      = normal.sample().to(self.device)
        action = torch.tanh(mean + std*z)
        
        action  = action.cpu()#.detach().cpu().numpy()
        return action[0]

    def save(self, appendix: str = ''):
        """
        Saves the current model parameters to file.
        :param appendix: An appendix to add to the file name of the model.
        """
        directory = os.path.dirname(__file__)
        path = os.path.join(directory, PolicyNetwork.model_directory_save, self.name + appendix)
        torch.save(self.state_dict(), path)