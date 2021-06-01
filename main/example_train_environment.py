import os
import numpy as np

from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment

# full documentation
# https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Python-API.md

dirname = os.path.dirname(__file__)
env_path = os.path.join(dirname, '../environment/Crawler.exe')

# pass the relative path to the unity build of the target domain
# filename set to None interacts directly with the editor
env = UnityEnvironment(file_name=env_path)

# returns the enviroment to it's default state
env.reset()

# get the name of the first behaviour in the domain
# since we only have one type of agent and no teams this is always the first and only behaviour
behaviour_name = list(env.behavior_specs)[0]
print('The name of the behaviour is:', behaviour_name)

for i in range(100):
    print('\nSTEP', i + 1)

    # get information about the state of the domain
    decisions, terminals = env.get_steps(behaviour_name)

    # decisions contains information about all the agents that requested an action since the last step
    # note that this must no be all the agents in the environment
    print('Number of agents requesting an action:', len(decisions))

    # decisions holds information about the observations made by each agent
    # obs is a list of the different obervations
    # each observation is an ndarray where the first dimension corresponds to the number of agents requesting actions
    print('Shape of first observation array:', decisions.obs[0].shape)

    # decitions also holds the rewards
    print('Rewards received since last step:', np.round_(decisions.reward, decimals=2))

    # terminals on the other hand holds mostly the same information as decisions
    # but for all the agents that have completed their episode
    print('Number of agents in a terminal state:', len(terminals))

    # we can use the information from above to feed it into the network to get actions and perform updates
    action = ActionTuple()

    # the action tuple contains the (20) continuous actions for each agent
    # the first dimension again represents the different agents requesting actions
    # we provide all ones which apparently turns the crawler into a weird ballerina
    action.add_continuous(np.ones((len(decisions), 20)))

    # to actually apply the actions to the agents set_actions can be used
    env.set_actions(behaviour_name, action)

    # step the environment to advance to the next step
    env.step()

# closes the connection to the environment
# closes the unity build window
env.close()
