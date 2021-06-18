from src.sac.sac_parameters import Parameters
from src.sac.sac_actor_critic import PolicyNetwork
from src.utils.domain import Domain
from src.utils.wrapper import CrawlerWrapper
import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# create a crawler environment and wrap it in the gym wrapper
env = Domain().environment()
env = CrawlerWrapper(env)

# define the hyper parameters
params = Parameters(env.observation_space_size, env.action_space_size)

inputs = env.observation_space_size
outputs = env.action_space_size

hidden_dim = 512
max_frames = params.max_frames
max_steps = params.max_steps

name = "testrun"

policy_net = PolicyNetwork(inputs, outputs, hidden_dim, name, device).to(device)

policy_net.load_state_dict(torch.load('../models/sac/temp/crawler2300', map_location=device))
policy_net.eval()

reward_mean = 0

for i in range(20):
    state = env.reset()
    reward_total = 0
    done = False

    while True:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = policy_net(state)
        action = policy_net.get_action(state).detach()
        state_next, reward, done, _ = env.step(action.squeeze().numpy())
        state = state_next.squeeze()

        reward_total += reward[0]
        env.render()
