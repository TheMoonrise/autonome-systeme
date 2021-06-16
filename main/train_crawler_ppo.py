import sys
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

# here we can maybe randomize some of the parameters and log the values to mlflow
# maybe we do actually want to allow passing a json file that holds the parameters
# so the options could be: random parameters, if nothing is passed; json values, if a file is passed
params.training_iterations = 10_000
params.clip = 0.2
params.epochs = 10
params.mini_batch_size = 28
params.influence_critic = 0.5
params.influence_entropy = 0.001
params.gamma = 0.9
params.lmbda = 1
params.trace = 32
params.learning_rate = 1e-5

# check for cuda support
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Training will be performed on {device}')

# create a model to train and optimizer
# if given as an commandline argument a pretrained model is used a starting point
use_pretrained_model = len(sys.argv) > 1
model_name = sys.argv[1] if use_pretrained_model else 'crawler'

model = ActorCriticCrawler(params, model_name).to(device)
if use_pretrained_model: model.load(device)

optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

# run the training loop
train = TrainAndEvaluate(env, model)
train.train(params, optimizer, device, 500)

# plot the results
plt.plot(train.performance)
plt.show()
