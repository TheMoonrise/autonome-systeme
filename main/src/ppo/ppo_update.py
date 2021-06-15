"""Provides functionality to update a model with given traces"""

import torch
import numpy as np

from .ppo_actor_critic import ActorCritic
from .ppo_parameters import Parameters


class Update:
    """
    Holds functionality to perform optimization steps from collected traces.
    """

    def __init__(self, params: Parameters, states, actions, probs, returns, advantages):
        """
        Initializes the updater.
        :param params: Parameters container holding training and update parameters.
        :param states: The state arrays of the traces.
        :param actions: The actions of the traces.
        :param probs: The action probabilities of the traces.
        :param returns: The state returns within the traces.
        :param advantages: The advantages for each trace.
        """
        self.params = params
        self.states = states
        self.actions = actions
        self.probs = probs
        self.returns = returns
        self.advantages = advantages

        self.batch_size = states.size(0)

    def _mini_batch(self):
        """
        Divides the batch into minibatches.
        :returns: A minibatch of the current trace data.
        """
        selection = np.random.randint(0, self.batch_size, self.params.mini_batch_size)

        states = self.states[selection, :]
        actions = self.actions[selection, :]
        probs = self.probs[selection, :]
        returns = self.returns[selection, :]
        advantages = self.advantages[selection, :]

        return states, actions, probs, returns, advantages

    def update(self, optimizer: torch.optim.Optimizer, model: ActorCritic):
        """
        Performs multiple optimization steps with the given optimizer.
        :param optimizer: The optimizer used for improving the model.
        :param model: The ppo actor critic model being optimized.
        """
        for _ in range(self.params.epochs):
            states, actions, probs_old, returns, advantages = self._mini_batch()

            states = states.reshape((-1, self.params.inputs))
            actions = actions.reshape((-1, self.params.outputs))

            dist, values = model(states)
            entropy = dist.entropy().mean()
            probs_new = dist.log_prob(actions)

            probs_new = probs_new.reshape(probs_old.shape)
            values = values.reshape((self.params.mini_batch_size, -1))

            ratio = (probs_new - probs_old).exp().mean(dim=2)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.params.clip, 1.0 + self.params.clip) * advantages

            objc_actor = torch.min(surr1, surr2).mean()
            loss_critic = (returns - values).pow(2).mean()

            loss = -(objc_actor - self.params.influence_critic * loss_critic + self.params.influence_entropy * entropy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
