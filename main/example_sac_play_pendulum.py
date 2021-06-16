from src.sac.sac_actor_critic import PolicyNetwork
from src.sac.sac_functions import NormalizedActions
import gym
import torch

hidden_dim = 256
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


env = NormalizedActions(gym.make("Pendulum-v0"))


action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]

name = "testrun"

policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, name, device).to(device)

policy_net.load_state_dict(torch.load('../models/sac/temp/pendulum4000', map_location=device))
policy_net.eval()

reward_mean = 0

for i in range(20):
    state = env.reset()
    reward_total = 0
    done = False

    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = policy_net(state)
        action = policy_net.get_action(state).detach()

        state_next, reward, done, _ = env.step(action.numpy())
        state = state_next.squeeze()

        reward_total += reward[0]
        env.render()

    reward_mean += reward_total
    print("Mean Reward Episode ", i, reward_mean / 20)
