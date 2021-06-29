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

# parse argument from cmd
parser = ArgumentParser(description='sac training crawler')
parser.add_argument('--fname', type=str, help='the name under which the trained model is stored', default="test_run")
parser.add_argument('--params', type=str, help='The parameter file for the model.')
parser.add_argument('--speed', type=float, help='The time scale of the simulation.', default=1)
args = parser.parse_args()

# create a crawler environment and wrap it in the gym wrapper
env = Domain().environment(time_scale=args.speed)
env = CrawlerWrapper(env)

# load environment variables
load_dotenv()

# define the hyper parameters
params = Parameters(env.observation_space_size, env.action_space_size, args.fname)
if args.params is not None: params.load(args.params)

inputs = env.observation_space_size
outputs = env.action_space_size

hidden_dim = 512
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

soft_q_net1 = SoftQNetwork(inputs, outputs, hidden_dim).to(device)
soft_q_net2 = SoftQNetwork(inputs, outputs, hidden_dim).to(device)

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
                action = policy_net.get_action(state).detach()
                next_state, reward, done, _ = env.step(action.numpy())
            else:
                action = np.random.uniform(low=np.nextafter(-1.0, 0.0), high=1.0, size=(10, 20))
                next_state, reward, done, _ = env.step(action)

            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += np.mean(reward)

            if len(replay_buffer) > batch_size:
                sac_update(batch_size, gamma, soft_tau, episode)

            # if episode % 10000 == 0:
            #     print('Epoch:{}, episode reward is {}'.format(episode, episode_reward))
            #     policy_net.save(str(episode))
            #     mlflow.pytorch.log_model(policy_net, str(episode))

            if done[0]:
                performance = episode_reward
                mlflow.log_metric('performance', performance, step=episode)
                mlflow.log_metric('episode length', step_count, step=episode)
                break

        if episode % 1000 == 0:
            print('Epoch:{}, episode reward is {}'.format(episode, episode_reward))
            path_to_current_model = policy_net.save(str(episode))
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
        # print(i, ":", reward_total)
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
    # state has size (128, 10, 158)
    # action has size (128, 10, 20)
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action = torch.FloatTensor(action).to(device)
    reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    # Passing data into the Soft Q network model
    predicted_q_value1 = soft_q_net1(state, action)
    predicted_q_value2 = soft_q_net2(state, action)

    # Calculate the predicted value of the SQN
    # predicted_value has size of torch.Size([128, 10, 1])
    predicted_value = value_net(state)
    # Return the next action and the entropies based on the policy update
    # new_action and log_prob have the size of 128, 10, 20
    new_action, log_prob, epsilon, mean, log_std = policy_net.evaluate(state)

# Training Q Function
    # target_value has torch.Size([128, 10, 1])
    target_value = target_value_net(next_state)
    # Calculate the actual value of the SQN
    # reward and done have torch.Size([128, 1, 10])
    # FIRST HACK: Swap 2nd and 3rd dimension
    reward = reward.transpose(1, 2)
    done = done.transpose(1, 2)

    # SECOND HACK: select only first element to make them one dimensional
    # target_q_value = reward[0][0][0] + (1 - done[0][0][0]) * gamma * target_value

    target_q_value = reward + (1 - done) * gamma * target_value
    # Calculate the loss between the predicted and the actual value of SQN
    # without HACK: this returns a warning bc predicted_q_value1 128, 10, 1 and target_q_value would have 128, 10, 10
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
    # predicted_new_q_value size 128, 10, 1
    # new_action size 128, 10, 20
    # state size 128, 10, 158
    # Choose minimum between model 1 and model 2 Q-value for agent
    predicted_new_q_value = torch.min(soft_q_net1(state, new_action), soft_q_net2(state, new_action))
    # target_value_func size 128, 10, 20
    # log_prob size 128, 10, 20
    target_value_func = predicted_new_q_value - log_prob
    # HACK: reshape predicted_value
    reshape_v = torch.zeros(batch_size, 10, outputs).to(device)
    predicted_value = predicted_value - reshape_v
    # without HACK: returns a warning bc predicted_value 128, 10, 1 and target_value_func 128, 10, 20
    value_loss = value_criterion(predicted_value, target_value_func.detach())

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    mlflow.log_metric('value loss', value_loss.item(), step=episode)

# Training Policy Function
    policy_loss = (log_prob - predicted_new_q_value).mean()

    # print("log_prob: ", log_prob[0][0])
    # print("predicted_new_q_value: ", predicted_new_q_value[0][0])
    # print("policy_loss: ", policy_loss)
    # difference = log_prob - predicted_new_q_value
    # print("difference: ", difference[0][0])

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    mlflow.log_metric('loss', policy_loss.item(), step=episode)

    # Soft update model parameters. θ_target = τ*θ_local + (1 - τ)*θ_target
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
