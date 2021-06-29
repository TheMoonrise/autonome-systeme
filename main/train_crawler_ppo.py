import mlflow
import os
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
parser.add_argument('--no-mlflow', help='Disables mlflow logging for the run.', action='store_true')

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
    params.mlflow = not args.no_mlflow

    # check for cuda support
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training will be performed on {device}')
    print(f"This training run uses the following parameters:\n{params.__dict__}")

    # create a model to train and optimizer
    # if given as an commandline argument a pretrained model is used a starting point
    use_pretrained_model = args.model is not None
    model_name = args.model if use_pretrained_model else 'crawler'

    model = ActorCriticCrawler(params, model_name).to(device)
    if use_pretrained_model: model.load(device)

    # load the optimizer state if available
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    opti_path = model.optimizer_path(is_save=False)

    if (os.path.exists(opti_path)): optimizer.load_state_dict(torch.load(opti_path, map_location=device))
    else: print('No optimizer state found for', model_name)

    # start mlflow run
    # if no run is active methods like mlflow.log_param will create a new run
    # a run is autometically closed when the with statement exits
    name_appendix = f'-{args.params.replace(".json", "").replace("_", "-")}' if args.params is not None else ''

    if params.mlflow:
        print('Starting mlflow run')
        mlflow.start_run(run_name='ppo' + name_appendix)
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

        if params.mlflow:
            mlflow.set_tag('error', e)

    # generate some graphics and save them to mlflow
    s_dir = os.path.join(model.model_directory(True), 'plots')
    os.makedirs(s_dir, exist_ok=True)
    plots = Plots(s_dir, 'ppo')

    plots.plot_performance(train.performance)
    plots.plot_moving_avg_performance(train.performance)

    if params.mlflow:
        mlflow.log_artifacts(s_dir)
        mlflow.end_run()
