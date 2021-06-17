import sys

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

# create a model and load the parameters
# use the first commandline argument for the model if given
model = ActorCriticCrawler(params, 'crawler-v1' if len(sys.argv) < 2 else sys.argv[1])
model.load()

# run the training loop
train = TrainAndEvaluate(env, model)

while True:
    result = train.evaluate(True)
    print(f"Performance (reward) {result:.2f}")
