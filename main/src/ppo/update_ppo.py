"""Provides functionality to update a model with given traces"""

import torch
import numpy as np

from .actor_critic_ppo import ActorCriticPPO


class UpdatePPO:
    """
    Holds functionality to perform optimization steps from collected traces.
    """

    def __init__(self, params, states, actions, probs, returns, advantages):
        """
        Initializes the updater.
        :param params: Dictionary holding training and update parameters.
        :param states: The state arrays of the traces.
        :param actions: The actions of the traces.
        :param probs: The action probabilities of the traces.
        :param returns: The state returns within the traces.
        :param advantages: The advantages for each trace.
        """
        assert 'ppo_clip' in params
        assert 'ppo_epochs' in params
        assert 'ppo_mini_batch_size' in params
        assert 'ppo_influence_critic' in params
        assert 'ppo_influence_entropy' in params

        self.clip = params['ppo_clip']
        self.epochs = params['ppo_epochs']
        self.mini_batch_size = params['ppo_mini_batch_size']
        self.influence_critic = params['ppo_influence_critic']
        self.influence_entropy = params['ppo_influence_entropy']

        self.states = states
        self.actions = actions
        self.probs = probs
        self.returns = returns
        self.advantages = advantages

        self.batch_size = states.size(0)

    def _batch_iterator(self):
        """
        Divides the batch into minibatches and yields iterations of them
        """
        for _ in range(self.batch_size // self.mini_batch_size):
            selection = np.random.randint(0, self.batch_size, self.mini_batch_size)

            states = self.states[selection, :]
            actions = self.actions[selection, :]
            probs = self.probs[selection, :]
            returns = self.returns[selection, :]
            advantages = self.advantages[selection, :]

            yield states, actions, probs, returns, advantages

    def update(self, optimizer: torch.optim.Optimizer, model: ActorCriticPPO):
        """
        Performs multiple optimization steps with the given optimizer.
        :param optimizer: The optimizer used for improving the model.
        :param model: The ppo actor critic model being optimized.
        """
        for _ in range(self.epochs):
            for states, actions, probs_old, returns, advantages in self._batch_iterator():
                dist, values = model(states)
                entropy = dist.entropy().mean()
                probs_new = dist.log_prob(actions)

                ratio = probs_new / probs_old
                # ratio = (probs_new - probs_old).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advantages

                loss_actor = -torch.min(surr1, surr2).mean()
                loss_critic = (returns - values).pow(2).mean()

                loss = self.influence_critic * loss_critic + loss_actor - self.influence_entropy * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
