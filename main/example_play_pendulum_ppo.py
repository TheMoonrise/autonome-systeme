import gym
import sys

from src.ppo.ppo_parameters import Parameters
from src.ppo.ppo_actor_critic_pendulum import ActorCriticPendulum
from src.ppo.ppo_training import TrainAndEvaluate

# prepare the open ai gym environment
env = gym.make('Pendulum-v0')

# define the hyper parameters
inputs = env.observation_space.shape[0]
outputs = env.action_space.shape[0]

params = Parameters(inputs, outputs)

# create a model and load the parameters
# use the first commandline argument for the model if given
model = ActorCriticPendulum(params, 'pendulum-v0' if len(sys.argv) < 2 else sys.argv[1])
model.load()

# run the evaluation loop indefinitely
train = TrainAndEvaluate(env, model)

while True:
    result = train.evaluate(True)
    print(f"Performance (reward) {result:.2f}")
