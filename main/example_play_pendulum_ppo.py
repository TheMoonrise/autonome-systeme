import gym
import sys

from src.ppo.ppo_parameters import Parameters
from src.ppo.ppo_actor_critic import ActorCritic
from src.ppo.ppo_training import ppo_evaluate

# prepare the open ai gym environment
env = gym.make('Pendulum-v0')

# define the hyper parameters
inputs = env.observation_space.shape[0]
outputs = env.action_space.shape[0]

params = Parameters(inputs, outputs)

# create a model and load the parameters
# use the first commandline argument for the model if given
model = ActorCritic(params)
model.load('pendulum-v0' if len(sys.argv) < 2 else sys.argv[1])

# run the evaluation loop indefinitely
while True:
    result = ppo_evaluate(env, model, 'cpu', 1, True)
    print(f"Performance (reward) {result:.2f}")
