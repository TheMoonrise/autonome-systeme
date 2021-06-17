import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import mlflow
import os
from dotenv import load_dotenv

from src.ppo.ppo_parameters import Parameters
from src.ppo.ppo_actor_critic_crawler import ActorCriticCrawler
from src.ppo.ppo_training import TrainAndEvaluate
from src.utils.domain import Domain
from src.utils.wrapper import CrawlerWrapper

parser = ArgumentParser(description='The argument mining prediction.')
parser.add_argument('--runs', type=int, help='how many times the training will be performed.', default=1)
parser.add_argument('--model-name', type=str, help='The name of the model to load.')
parser.add_argument('--params', type=str, help='The parameter file for the model.')
args = parser.parse_args()

# create a crawler environment and wrap it in the gym wrapper
env = Domain().environment()
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

    # create a model to train and optimizer
    # if given as an commandline argument a pretrained model is used a starting point
    use_pretrained_model = args.model_name is not None
    model_name = args.model_name if use_pretrained_model else 'crawler'

    model = ActorCriticCrawler(params, model_name).to(device)
    if use_pretrained_model: model.load(device)

    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

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
        mlflow.log_param('training_iterations', params.training_iterations)
        mlflow.log_param('clip', params.clip)
        mlflow.log_param('epochs', params.epochs)
        mlflow.log_param('mini_batch_size', params.mini_batch_size)
        mlflow.log_param('influence_critic', params.influence_critic)
        mlflow.log_param('influence_entropy', params.influence_entropy)
        mlflow.log_param('gamma', params.gamma)
        mlflow.log_param('lmbda', params.lmbda)
        mlflow.log_param('trace', params.trace)
        mlflow.log_param('learning_rate', params.learning_rate)

        # run the training loop
        train = TrainAndEvaluate(env, model)
        train.train(params, optimizer, device, 500)

        # plot the results
        figure = plt.plot(train.performance)
        
        # plt.savefig(fig_path)

        # log plot
        # mlflow.log_artifact(fig_path)

        plt.show()
        