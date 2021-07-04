import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import mlflow
from dotenv import load_dotenv
from argparse import ArgumentParser

from src.sac.sac_parameters import Parameters
from src.sac.sac_actor_critic_crawler import ValueNetwork, SoftQNetwork, PolicyNetwork
from src.sac.sac_functions import ReplayBuffer
from src.utils.domain import Domain
from src.utils.wrapper import CrawlerWrapper

# parse arguments from cmd
parser = ArgumentParser(description='sac training crawler')
parser.add_argument('--fname', type=str, help='the name under which the trained model is stored', default="test_run")
parser.add_argument('--runs', type=int, help='how many times the training will be performed.', default=1)
parser.add_argument('--model', type=str, help='The name of the model to load.')
parser.add_argument('--params', type=str, help='The parameter file for the model.')
parser.add_argument('--tag', type=str, help='An additional tag for identifying this run.')

parser.add_argument('--speed', type=float, help='Define the speed at which the simulation runs.', default=1)
parser.add_argument('--quality', type=int, help='Define the quality of the simulation.', default=0)
parser.add_argument('--slipperiness', type=float, help='Define how slippery the ground is [0, 1]', default=0)
parser.add_argument('--steepness', type=float, help='Define how steep and uneven the terrain is [0, 1]', default=0)
parser.add_argument('--hue', type=float, help='Defines the color hue of the crawler [0, 360]', default=50)

parser.add_argument('--no-window', help='Hides the simulation window.', action='store_true')
parser.add_argument('--no-mlflow', help='Disables mlflow logging for the run.', action='store_true')

args = parser.parse_args()

# create a crawler environment and wrap it in the gym wrapper
env = Domain().environment(args.speed, args.quality, args.no_window, args.slipperiness, args.steepness, args.hue)
env = CrawlerWrapper(env)

# load environment variables if training should be logged on the mlflow server
if not args.no_mlflow:
    load_dotenv()

# set the hyper parameters
params = Parameters(env.observation_space_size, env.action_space_size, args.fname, args.speed)
if args.params is not None: params.load(args.params)

inputs = env.observation_space_size
outputs = env.action_space_size

hidden_dim = params.hidden_dim
value_lr = params.value_lr
soft_q_lr = params.soft_q_lr
policy_lr = params.policy_lr
replay_buffer_size = params.replay_buffer_size
batch_size = params.batch_size
soft_tau = params.soft_tau
gamma = params.gamma
max_episodes = params.max_episodes
initial_exploration = params.initial_exploration
max_steps = params.max_steps

# check for cuda support
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Training will be performed on {device}')

# create a model to train and optimizer
model_name = 'crawler'

value_net = ValueNetwork(inputs, hidden_dim).to(device)
target_value_net = ValueNetwork(inputs, hidden_dim).to(device)

# Two Q-functions significantly speed up training by reducing overestimation bias according to paper
soft_q_net1 = SoftQNetwork(inputs, outputs, hidden_dim).to(device)
soft_q_net2 = SoftQNetwork(inputs, outputs, hidden_dim).to(device)

# main actor network
policy_net = PolicyNetwork(inputs, outputs, hidden_dim, model_name, device, params).to(device)

"""
value_net: PyTorch model (weights will be copied from)
target_value_net: PyTorch model (weights will be copied to)
"""
for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)

# specify the standard for the optimization
# The Mean Squared Error (MSE) computes the average of the squared differences
# between actual values and predicted values
value_criterion = nn.MSELoss()
soft_q_criterion1 = nn.MSELoss()
soft_q_criterion2 = nn.MSELoss()

# create an optimizer for the value, soft q and policy networks
value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)
soft_q_optimizer1 = optim.Adam(soft_q_net1.parameters(), lr=soft_q_lr)
soft_q_optimizer2 = optim.Adam(soft_q_net2.parameters(), lr=soft_q_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)

# Replay buffer to add gathered samples to.
replay_buffer = ReplayBuffer(replay_buffer_size)


def sac_train():
    """
    Initiates training loop.
    """
    episode = 0
    rewards = []

    while episode < max_episodes:
        state = env.reset()
        episode_reward = 0

        for step_count in range(max_steps):
            if episode > initial_exploration:
                # observe state and select action
                action = policy_net.get_action(state).detach()
                # execute selected action in the environment
                next_state, reward, done, _ = env.step(action.numpy())
            else:
                # behave as a random agent for the initial exploration phase
                action = np.random.uniform(low=np.nextafter(-1.0, 0.0), high=1.0, size=(10, 20))
                next_state, reward, done, _ = env.step(action)

            # collecting experience from the environment with the current policy by storing into replay buffer
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += np.mean(reward)

            if len(replay_buffer) > batch_size:
                sac_update(batch_size, gamma, soft_tau, episode)

            if done[0]:
                performance = episode_reward
                mlflow.log_metric('performance', performance, step=episode)
                mlflow.log_metric('episode length', step_count, step=episode)
                break

        if episode % 1000 == 0:
            # save trained models every 1000 (10.000) episodes
            print('Epoch:{}, episode reward is {}'.format(episode, episode_reward))
            path_to_current_model = policy_net.save(str(episode))
            if episode % 10000 == 0:
                mlflow.pytorch.log_model(policy_net, str(episode))

        rewards.append(episode_reward)
        episode += 1

    # evaluate trained model
    policy_net.load_state_dict(torch.load(path_to_current_model, map_location=device))
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
            iteration_done = done.item(0)

            reward_total += reward[0]
            env.render()

        mlflow.log_metric('reward test episode', reward_total, step=i)
        all_rewards.append(reward_total)

    reward_mean = sum(all_rewards) / len(all_rewards)
    mlflow.log_param('mean test reward', reward_mean)
    print("Mean Reward after", number_iterations, "iterations:", reward_mean)


def sac_update(batch_size, gamma, soft_tau, episode):
    """
    method that updates the two q functions, the value function and the policy function
    :param batch_size: batch size that is used for the update
    :param gamma: discount factor applied to the rewards
    :param soft_tau: soft update coefficient for the target network
    """
    # randomly sample a batch of transitions from the replay buffer
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action = torch.FloatTensor(action).to(device)
    reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    # Passing data into the Soft Q network model
    predicted_q_value1 = soft_q_net1(state, action)
    predicted_q_value2 = soft_q_net2(state, action)

    # predicted_value: prediction of our value network
    predicted_value = value_net(state)

    # Return the next action and the entropies based on the policy update
    new_action, log_prob, epsilon, mean, log_std = policy_net.evaluate(state)

# Training Q Function
    target_value = target_value_net(next_state)
    reward = reward.transpose(1, 2)
    done = done.transpose(1, 2)

    # compute targets for the Q-functions: yt(r,s′,d)=r+γ(minj=1,2Qϕtarg,j(s′,a~′)−αlogπθ(a~′|s′))
    target_q_value = reward + (1 - done) * gamma * target_value

    # Calculate the loss between the predicted and the actual value of SQN
    q_value_loss1 = soft_q_criterion1(predicted_q_value1, target_q_value.detach())

    q_value_loss2 = soft_q_criterion2(predicted_q_value2, target_q_value.detach())

    # Zero the gradient buffers
    soft_q_optimizer1.zero_grad()

    # Propagating the loss back
    q_value_loss1.backward()
    soft_q_optimizer1.step()
    soft_q_optimizer2.zero_grad()
    q_value_loss2.backward()
    soft_q_optimizer2.step()

    mlflow.log_metric('q1 loss', q_value_loss1.item(), step=episode)
    mlflow.log_metric('q2 loss', q_value_loss2.item(), step=episode)

# Training Value Function
    # Choose minimum of Q-functions for the value gradient and policy gradient
    predicted_new_q_value = torch.min(soft_q_net1(state, new_action), soft_q_net2(state, new_action))

    # log_prob: entropy of the policy function π (measured here by the negative log of the policy function)
    # See equation 6 from [1]
    target_value_func = predicted_new_q_value - log_prob
    reshape_v = torch.zeros(batch_size, 10, outputs).to(device)
    predicted_value = predicted_value - reshape_v

    value_loss = value_criterion(predicted_value, target_value_func.detach())

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    mlflow.log_metric('value loss', value_loss.item(), step=episode)

# Training Policy Function
    # policy optimization with equation 10 from [1]: maxθE(s∼D,ξ∼N)[minj=1,2Qϕj(s,a~θ(s,ξ))−αlogπθ(a~θ(s,ξ)|s)]
    policy_loss = (log_prob - predicted_new_q_value).mean()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    mlflow.log_metric('loss', policy_loss.item(), step=episode)

    # Soft update model parameters of the value function. θ_target = τ*θ_local + (1 - τ)*θ_target
    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )


# start mlflow run
# if no run is active methods like mlflow.log_param will create a new run
# a run is autometically closed when the with statement exits
with mlflow.start_run(run_name=params.file_name) as run:
    print('Running mlflow.')

    # for returning informaton about the run
    client = mlflow.tracking.MlflowClient()
    print('Currently active run: ', client.get_run(mlflow.active_run().info.run_id).data)

    # log params, key and value are both strings
    params.log_to_mlflow()

    # Start Training
    sac_train()
