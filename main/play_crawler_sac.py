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

name = "testrun"
file_name = "default"

# define the hyper parameters
params = Parameters(env.observation_space_size, env.action_space_size, file_name)

inputs = env.observation_space_size
outputs = env.action_space_size

hidden_dim = 512
max_frames = params.max_episodes
max_steps = params.max_steps

policy_net = PolicyNetwork(inputs, outputs, hidden_dim, name, device).to(device)

policy_net.load_state_dict(torch.load('../models/sac/temp/crawler300000', map_location=device))
policy_net.eval()

all_rewards = []

number_iterations = 100

for i in range(number_iterations):
    state = env.reset()
    reward_total = 0

    done = False
    iteration_done = False

    while not iteration_done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = policy_net(state)
        action = policy_net.get_action(state).detach()
        state_next, reward, done, _ = env.step(action.squeeze().numpy())
        state = state_next.squeeze()
        iteration_done=done.item(0)

        reward_total += reward[0]
        env.render()
    # print(i, ":", reward_total)
    all_rewards.append(reward_total)
reward_mean = sum(all_rewards) / len(all_rewards)
print("Mean Reward after", number_iterations, "iterations:", reward_mean)
