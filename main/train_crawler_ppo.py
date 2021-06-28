import mlflow
import tempfile
import torch
import torch.optim as optim

from dotenv import load_dotenv
from argparse import ArgumentParser

from src.ppo.ppo_parameters import Parameters
from src.ppo.ppo_actor_critic_crawler import ActorCriticCrawler
from src.ppo.ppo_training import TrainAndEvaluate
from src.utils.domain import Domain
from src.utils.wrapper import CrawlerWrapper
from src.plots.plots import Plots

parser = ArgumentParser(description='The argument mining prediction.')

parser.add_argument('--runs', type=int, help='how many times the training will be performed.', default=1)
parser.add_argument('--model', type=str, help='The name of the model to load.')
parser.add_argument('--params', type=str, help='The parameter file for the model.')
parser.add_argument('--tag', type=str, help='An additional tag for identifying this run.')

parser.add_argument('--speed', type=float, help='Define the speed at which the simulation runs.', default=1)
parser.add_argument('--quality', type=float, help='Define the quality of the physics simulation', default=1)
parser.add_argument('--no-window', help='Hides the simulation window.', action='store_true')

args = parser.parse_args()

# create a crawler environment and wrap it in the gym wrapper
env = Domain().environment(args.speed, args.quality, args.no_window)
env = CrawlerWrapper(env)

# load environment variables
load_dotenv()

for run in range(args.runs):
    # define the hyper parameters
    params = Parameters(env.observation_space_size, env.action_space_size)
    if args.params is not None: params.load(args.params)

    # check for cuda support
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training will be performed on {device}')
    print(f"This training run uses the following parameters: {params.__dict__}")

    # create a model to train and optimizer
    # if given as an commandline argument a pretrained model is used a starting point
    use_pretrained_model = args.model is not None
    model_name = args.model if use_pretrained_model else 'crawler'

    model = ActorCriticCrawler(params, model_name).to(device)
    if use_pretrained_model: model.load(device)

    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # start mlflow run
    # if no run is active methods like mlflow.log_param will create a new run
    # a run is autometically closed when the with statement exits
    name_appendix = f'-{args.params.replace(".json", "").replace("_", "-")}' if args.params is not None else ''

    with mlflow.start_run(run_name='ppo' + name_appendix) as run:
        print('Starting mlflow run')
        params.log_to_mlflow()

        if args.model is not None: mlflow.set_tag('parent model', model.name)
        if args.tag is not None: mlflow.set_tag('tag', args.tag)

        try:
            # run the training loop
            train = TrainAndEvaluate(env, model)
            train.train(params, optimizer, device, 1000)

        except Exception as e:
            print('Training ended prematurely')
            print(e)

        # generate some graphics and save them to mlflow
        with tempfile.TemporaryDirectory() as dir:
            plots = Plots(dir, 'ppo')
            plots.plot_performance(train.performance)

            mlflow.log_artifacts(dir)
