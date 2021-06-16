"""Utility to use the unity crawler domain as openai gym environment"""

import numpy as np

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple


class CrawlerWrapper:
    """
    This wrapper can be used to allow the uniy crawler domain to be used as an openai gym environment.
    The wrapper, however, provides all outputs as ndarrays to allow for simultaenous training of multiple agents.
    This is different to regular openai environments where most outputs are single values.
    Therefor this is not a drop in replacement but requires some adjustment of training code.
    """

    def __init__(self, crawler_env: UnityEnvironment):
        """
        Initializes the wrapper environment.
        :param crawler_env: The crawler unity environment wrapped.
        """
        self.env = crawler_env
        self.env.reset()

        self.behaviour_name = list(self.env.behavior_specs)[0]

        # the agent count is the number of agents requesting actions in the first step
        self.num_agents = len(self.env.get_steps(self.behaviour_name)[0])
        self.action_mask = None

        behaviour_specs = self.env.behavior_specs[self.behaviour_name]

        # use these values to initialize the neural networks
        # this is the alternative to env.observation_space.shape[0] used on regular gym environments
        self.observation_space_size = sum([x.shape[0] for x in behaviour_specs.observation_specs])
        self.action_space_size = behaviour_specs.action_spec.continuous_size
        

    def render(self):
        """
        This function is only to meet the gym interface.
        """

    def close(self):
        """
        Closes the unity environment.
        This function has no analog in the gym environments.
        """
        self.env.close()

    def reset(self):
        """
        Works just as the reset function in openai gym environments.
        Restores the default state of the environment.
        :returns: The initial state of the environment.
        """
        self.env.reset()
        state, _, _ = self._sample_state()
        return state

    def step(self, actions: np.ndarray):
        """
        Advances the environment using the given actions.
        :param actions: The actions to be performed for each agent.
        This is a ndarray of the shape (num_agents, action_space_size).
        :returns: A ndarray of state containing all agent observations if available.
        This has the shape (num_agents, observation_space_size).
        :returns: A ndarray of rewards collected by all agents.
        This has the shape (num_agents,).
        :returns: A ndarray of boolean flags indicating if the agent is in a terminal state.
        This has the shape (num_agents,).
        :returns: Additional information about the state.
        This is not yet implemented and mainly meant to achieve compatability with the gym environment step function.
        """
        action_tuple = ActionTuple()
        action_tuple.add_continuous(actions[self.action_mask])
        self.env.set_actions(self.behaviour_name, action_tuple)
        self.env.step()

        state, rewards, done = self._sample_state()
        return state, rewards, done, None

    def _sample_state(self):
        """
        Collects training information about the environment in the current step.
        Because not every step requires agent action input this can cover multiple steps.
        The environment is advanced until agent input is requested.
        Training data is accumulated while this happens.
        :returns: A ndarray of state containing all agent observations if available.
        :returns: A ndarray of rewards collected by all agents.
        :returns: A ndarray of boolean flags indicating if the agent is in a terminal state.
        """
        done = np.zeros((self.num_agents,), dtype=bool)
        rewards = np.zeros((self.num_agents,), dtype=float)
        state = np.zeros((self.num_agents, self.observation_space_size), dtype=float)

        while True:
            # get information about the state of the domain
            decisions, terminals = self.env.get_steps(self.behaviour_name)

            # combine different observations into one array
            observations_decisions = np.concatenate(decisions.obs, axis=1)
            observations_terminals = np.concatenate(terminals.obs, axis=1)

            # update the state using the latest obeservations from acting and terminated agents
            state[decisions.agent_id] = observations_decisions
            state[terminals.agent_id] = observations_terminals

            # mark agents that are featured in a terminal state as done
            done[terminals.agent_id] = True

            # accumulate rewards for all agents
            rewards[decisions.agent_id] += decisions.reward
            rewards[terminals.agent_id] += terminals.reward

            # the agent ids that require actions function as the mask
            self.action_mask = decisions.agent_id

            # step the environment to advance to the next step
            # check if any agent is requiring an action in the current step
            # if that is not the case advance the simulation and accumulate the experience
            actions_requested = len(decisions) > 0
            if actions_requested: break
            self.env.step()

        return state, rewards, done