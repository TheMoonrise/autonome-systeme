import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import mlflow

from src.sac.sac_parameters import Parameters
from src.sac.sac_actor_critic_crawler import ValueNetwork, SoftQNetwork, PolicyNetwork
from src.sac.sac_functions import ReplayBuffer
from src.utils.domain import Domain
from src.utils.wrapper import CrawlerWrapper

# create a crawler environment and wrap it in the gym wrapper
env = Domain().environment()
env = CrawlerWrapper(env)

# define the hyper parameters
params = Parameters(env.observation_space_size, env.action_space_size)

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
max_frames = params.max_frames
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

policy_net = PolicyNetwork(inputs, outputs, hidden_dim, model_name, device).to(device)

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
    frame_idx = 0
    rewards = []

    while frame_idx < max_frames:
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            if frame_idx > 1000:
                action = policy_net.get_action(state).detach()
                next_state, reward, done, _ = env.step(action.numpy())
            else:
                action = np.random.uniform(low=np.nextafter(-1.0, 0.0), high=1.0, size=(10, 20))
                next_state, reward, done, _ = env.step(action)

            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            frame_idx += 1

            if len(replay_buffer) > batch_size:
                sac_update(batch_size, gamma, soft_tau)

            if frame_idx % 100 == 0:
                print('Epoch:{}, episode reward is {}'.format(frame_idx, episode_reward))
                policy_net.save(f'{frame_idx}')
                # plot(frame_idx, rewards)

            if done[0]:
                break

        rewards.append(episode_reward)


def sac_update(batch_size, gamma, soft_tau):
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
    reshape_v = torch.zeros(128, 10, 20).to(device)
    predicted_value = predicted_value - reshape_v
    # without HACK: returns a warning bc predicted_value 128, 10, 1 and target_value_func 128, 10, 20
    value_loss = value_criterion(predicted_value, target_value_func.detach())

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

# Training Policy Function
    policy_loss = (log_prob - predicted_new_q_value).mean()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    # Soft update model parameters. θ_target = τ*θ_local + (1 - τ)*θ_target
    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )


# start mlflow run
# if no run is active methods like mlflow.log_param will create a new run
# a run is autometically closed when the with statement exits
with mlflow.start_run() as run:
    print('Running mlflow.')

    # for returning informaton about the run
    client = mlflow.tracking.MlflowClient()
    print('Currently active run: ', client.get_run(mlflow.active_run().info.run_id).data)

    print('Tracking server: ', mlflow.tracking.get_tracking_uri())

    # creating an mlflow experiment
    # exp_id = mlflow.create_experiment('PPO')
    # print('Experiment id: ', exp_id)

    # log params, key and value are both strings
    mlflow.log_param('training iterations', params.max_frames)
    mlflow.log_param('clip', params.clip)
    mlflow.log_param('epochs', params.epochs)
    mlflow.log_param('batch size', params.batch_size)
    mlflow.log_param('influence critic', params.influence_critic)
    mlflow.log_param('influence entropy', params.influence_entropy)
    mlflow.log_param('gamma', params.gamma)
    mlflow.log_param('lambda', params.lmbda)
    mlflow.log_param('trace', params.trace)
    mlflow.log_param('learning rate', params.learning_rate)

# Start Training
sac_train()
