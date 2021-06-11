import gym
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim

from IPython.display import clear_output
import matplotlib.pyplot as plt

from pathlib import Path

from src.sac.sac_actor_critic import ValueNetwork, SoftQNetwork, PolicyNetwork
from src.sac.sac_functions import NormalizedActions, ReplayBuffer
from src.sac.sac_parameters import Parameters

model_name = 'pendulum'

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()

############################ INITALIZATIONS ################################

env = NormalizedActions(gym.make("Pendulum-v0"))

action_dim = env.action_space.shape[0]
state_dim  = env.observation_space.shape[0]

params = Parameters(state_dim, action_dim)

# network parameters
hidden_dim = params.hidden_dim
inputs = params.inputs
outputs = params.outputs
hidden_dim = params.hidden_dim

# learning and update parameters
value_lr  = params.value_lr
soft_q_lr = params.soft_q_lr
policy_lr = params.policy_lr
soft_tau = params.soft_tau
gamma = params.gamma

# training parameters
replay_buffer_size = params.replay_buffer_size
batch_size = params.batch_size
max_frames = params.max_frames
max_steps = params.max_steps

value_net        = ValueNetwork(inputs, hidden_dim).to(device)
target_value_net = ValueNetwork(inputs, hidden_dim).to(device)

soft_q_net1 = SoftQNetwork(inputs, outputs, hidden_dim).to(device)
soft_q_net2 = SoftQNetwork(inputs, outputs, hidden_dim).to(device)
policy_net = PolicyNetwork(inputs, outputs, hidden_dim, model_name, device).to(device)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)
    
value_criterion  = nn.MSELoss()
soft_q_criterion1 = nn.MSELoss()
soft_q_criterion2 = nn.MSELoss()


value_optimizer  = optim.Adam(value_net.parameters(), lr=value_lr)
soft_q_optimizer1 = optim.Adam(soft_q_net1.parameters(), lr=soft_q_lr)
soft_q_optimizer2 = optim.Adam(soft_q_net2.parameters(), lr=soft_q_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)

replay_buffer = ReplayBuffer(replay_buffer_size)

####################################################################################

def sac_train():
    """
    Initiates training loop.
    """
    frame_idx   = 0
    rewards     = []

    while frame_idx < max_frames:
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            if frame_idx >1000:
                action = policy_net.get_action(state).detach()
                next_state, reward, done, _ = env.step(action.numpy())
            else:
                action = env.action_space.sample()
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
            
            if done:
                break
            
        rewards.append(episode_reward)

def sac_update(batch_size,gamma,soft_tau):
    """
    TODO: add comment
    """
    
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action     = torch.FloatTensor(action).to(device)
    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    predicted_q_value1 = soft_q_net1(state, action)
    predicted_q_value2 = soft_q_net2(state, action)
    predicted_value    = value_net(state)
    new_action, log_prob, epsilon, mean, log_std = policy_net.evaluate(state)
    
# Training Q Function
    target_value = target_value_net(next_state)
    target_q_value = reward + (1 - done) * gamma * target_value
    q_value_loss1 = soft_q_criterion1(predicted_q_value1, target_q_value.detach())
    q_value_loss2 = soft_q_criterion2(predicted_q_value2, target_q_value.detach())

    soft_q_optimizer1.zero_grad()
    q_value_loss1.backward()
    soft_q_optimizer1.step()
    soft_q_optimizer2.zero_grad()
    q_value_loss2.backward()
    soft_q_optimizer2.step()    
# Training Value Function
    predicted_new_q_value = torch.min(soft_q_net1(state, new_action),soft_q_net2(state, new_action))
    target_value_func = predicted_new_q_value - log_prob
    value_loss = value_criterion(predicted_value, target_value_func.detach())
    
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()
# Training Policy Function
    policy_loss = (log_prob - predicted_new_q_value).mean()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()
    
    #Soft update model parameters. θ_target = τ*θ_local + (1 - τ)*θ_target
    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )

## Start Training
sac_train()
