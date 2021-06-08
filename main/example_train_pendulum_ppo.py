import gym
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from src.ppo.ppo_parameters import Parameters
from src.ppo.ppo_actor_critic_pendulum import ActorCriticPendulum
from src.ppo.ppo_training import TrainAndEvaluate

# prepare the open ai gym environment
env = gym.make('Pendulum-v0')

# define the hyper parameters
inputs = env.observation_space.shape[0]
outputs = env.action_space.shape[0]

params = Parameters(inputs, outputs)

params.training_iterations = 100_000
params.clip = 0.2
params.epochs = 10
params.mini_batch_size = 25
params.influence_critic = 0.5
params.influence_entropy = 0.001
params.gamma = 0.9
params.lmbda = 1
params.trace = 35
params.learning_rate = 1e-5

# check for cuda support
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Training will be performed on {device}')

# create a model to train and optimizer
model = ActorCriticPendulum(params, 'pendulum').to(device)
optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

# run the training loop
train = TrainAndEvaluate(env, model)
train.train(params, optimizer, device, 1000)

# plot the results
plt.plot(train.performance)
plt.show()
