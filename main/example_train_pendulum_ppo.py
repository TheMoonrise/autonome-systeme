import gym
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from src.ppo.ppo_parameters import Parameters
from src.ppo.ppo_actor_critic import ActorCritic
from src.ppo.ppo_training import ppo_train, ppo_evaluate

# prepare the open ai gym environment
env = gym.make('Pendulum-v0')
env_evaluation = gym.make('Pendulum-v0')

# define the hyper parameters
inputs = env.observation_space.shape[0]
outputs = env.action_space.shape[0]

params = Parameters(inputs, outputs)
params.training_iterations = 2500

# check for cuda support
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Training will use {device}')

# create a model to train and optimizer
model = ActorCritic(params).to(device)
optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

# define an evaluation function run during the training
performance = []


def f_evaluate():
    result = ppo_evaluate(env_evaluation, model, device, 10, True)
    performance.append(result)
    print(f"Average performance (reward) {result:.3f}")


# run the training loop
ppo_train(env, params, model, optimizer, device, f_evaluate)

# plot the results
plt.plot(performance)
plt.show()
