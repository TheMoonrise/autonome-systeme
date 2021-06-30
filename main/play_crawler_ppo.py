from argparse import ArgumentParser

from src.ppo.ppo_parameters import Parameters
from src.ppo.ppo_actor_critic_crawler import ActorCriticCrawler
from src.ppo.ppo_training import TrainAndEvaluate
from src.utils.domain import Domain
from src.utils.wrapper import CrawlerWrapper

parser = ArgumentParser(description='The argument mining prediction.')

parser.add_argument('--model', type=str, help='The name of the model to load.', default='crawler-v1')
parser.add_argument('--params', type=str, help='The parameter file for the model.')
parser.add_argument('--speed', type=float, help='Define the speed at which the simulation runs.', default=1)
parser.add_argument('--quality', type=int, help='Define the quality of the simulation.', default=0)

parser.add_argument('--slipperiness', type=float, help='Define how slippery the ground is [0, 1]', default=0)
parser.add_argument('--steepness', type=float, help='Define how steep and uneven the terrain is [0, 1]', default=0)

args = parser.parse_args()

# create a crawler environment and wrap it in the gym wrapper
env = Domain().environment(args.speed, args.quality, slipperiness=args.slipperiness, steepness=args.steepness)
env = CrawlerWrapper(env)

# define the hyper parameters
params = Parameters(env.observation_space_size, env.action_space_size)
if args.params is not None: params.load(args.params)

# create a model and load the parameters
# use the first commandline argument for the model if given
model = ActorCriticCrawler(params, args.model)
model.load()

# run the training loop
train = TrainAndEvaluate(env, model)

while True:
    result = train.evaluate(True)
    print(f"Performance (reward) {result:.2f}")
