import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from src.ppo.ppo_parameters import Parameters
from src.ppo.ppo_actor_critic_crawler import ActorCriticCrawler
from src.ppo.ppo_training import TrainAndEvaluate
from src.utils.domain import Domain
from src.utils.wrapper import CrawlerWrapper

# create a crawler environment and wrap it in the gym wrapper
env = Domain().environment()
env = CrawlerWrapper(env)

# define the hyper parameters
params = Parameters(env.observation_space_size, env.action_space_size)

params.training_iterations = 10_000
params.clip = 0.2
params.epochs = 10
params.mini_batch_size = 20
params.influence_critic = 0.5
params.influence_entropy = 0.001
params.gamma = 0.9
params.lmbda = 1
params.trace = 30
params.learning_rate = 1e-5

# check for cuda support
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Training will be performed on {device}')

# create a model to train and optimizer
model = ActorCriticCrawler(params, 'crawler').to(device)
optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

# run the training loop
train = TrainAndEvaluate(env, model)
train.train(params, optimizer, device, 1000)

# plot the results
plt.plot(train.performance)
plt.show()
