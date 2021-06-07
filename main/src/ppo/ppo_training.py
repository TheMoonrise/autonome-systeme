"""Handles training and evaluation of ppo networks"""

import torch
import numpy as np

from gym import Env
from torch.optim import Optimizer

from .ppo_parameters import Parameters
from .ppo_actor_critic import ActorCritic
from .ppo_update import Update
from .ppo_functions import ppo_returns


class TrainAndEvaluate():
    """
    Holds functionality for training, evaluating and visualizing ppo networks.
    """

    def __init__(self, env: Env, model: ActorCritic):
        """
        Initiates the training object.
        :param env: The environment to train or evaluate on.
        :param model: The ppo model used for training or evaluation.
        """
        self.env = env
        self.state = env.reset()
        self.model = model

        # training trace data collection
        self.states = []
        self.actions = []
        self.probs = []
        self.values = []

        self.rewards = []
        self.masks = []

        self.done = [False]

        # model performance tracking
        self.performance = []
        self.performance_counter = 0

    def train(self, params: Parameters, optimizer: Optimizer, device: str, save_interval: int = -1):
        """
        Performs a ppo training loop on the given model and environment.
        :param params: The parameters used for the training.
        :param optimizer: The optimizer configured for the model and used for improving based on collected experience.
        :param device: String property naming the device used for training.
        :param save_interval: The interval at which to save the current model.
        """
        print("Begin training")
        self.state = self.env.reset()

        for i in range(1, params.training_iterations + 1):
            performance = np.average(self.performance[-10:]) if self.performance else 0
            print(f"\rIteration, Epoch, Performance [ {i:^5} | {len(self.performance):^5} | {performance:^5.0f} ]", end='')

            self._clear_trace()
            self._collect_trace(params, device)

            # compute the discounted returns for each state in the trace
            state_next = torch.FloatTensor(self.state).to(device)
            if len(state_next.shape) < 2: state_next = state_next.unsqueeze(0)

            _, value_next = self.model(state_next)
            returns = ppo_returns(params, self.rewards, self.masks, self.values, value_next.squeeze(1))

            # combine the trace in tensors and update the model
            states = torch.stack(self.states)
            actions = torch.stack(self.actions)
            probs = torch.stack(self.probs).detach()
            values = torch.stack(self.values).detach()

            returns = torch.stack(returns).detach()
            advantages = returns - values

            update = Update(params, states, actions, probs, returns, advantages)
            update.update(optimizer, self.model)

            if save_interval > 0 and (i % save_interval == 0 or i == params.training_iterations):
                appendix = f'[{i:0>4}({performance:.0f})]'
                self.model.save(appendix)

    def evaluate(self, render: bool):
        """
        Runs an complete episode on the given environment for evaluation.
        For evaluation only the first agent is considered.
        :param render: Whether the environment should be rendered.
        :returns: The total reward collected over the evaluation.
        """
        self.state = self.env.reset()
        self.performance_counter = 0
        self.done = [False]

        while not self.done[0]:
            self.state = torch.FloatTensor(self.state)
            if len(self.state.shape) < 2: self.state = self.state.unsqueeze(0)

            dist, _ = self.model(self.state)
            action = dist.sample().cpu().numpy()

            state_next, reward, self.done, _ = self.env.step(action)
            self.state = state_next.squeeze()

            if isinstance(self.done, bool): self.done = [self.done]

            self.performance_counter += reward[0]
            if render: self.env.render()

        self.performance.append(self.performance_counter)
        return self.performance_counter

    def _collect_trace(self, params: Parameters, device: str):
        """
        Samples a trace of training data from the environment
        :param params: The parameters used for the training.
        :param device: String property naming the device used for training.
        """
        for _ in range(params.trace):
            if all(self.done): self.state = self.env.reset()
            self.state = torch.FloatTensor(self.state).to(device)

            # pendulum environment provides only a single state array
            if len(self.state.shape) < 2: self.state = self.state.unsqueeze(0)
            self.states.append(self.state)

            dist, value = self.model(self.state)
            self.values.append(value.squeeze(1))

            action = dist.sample()
            self.actions.append(action)

            state_next, rewards, self.done, _ = self.env.step(action.cpu().numpy())
            self.state = state_next.squeeze()

            # pendulum environment provides only a single bool
            if isinstance(self.done, bool): self.done = [self.done]

            self.rewards.append(torch.FloatTensor(rewards).to(device))
            self.masks.append(torch.FloatTensor(self.done).mul(-1).add(1).to(device))

            prob = dist.log_prob(action)
            self.probs.append(prob)

            # update performance tracking for the first agent
            self.performance_counter += rewards[0]

            if self.done[0]:
                self.performance.append(self.performance_counter)
                self.performance_counter = 0

    def _clear_trace(self):
        """
        Removes the current trace data.
        """
        self.states.clear()
        self.actions.clear()
        self.probs.clear()
        self.values.clear()

        self.rewards.clear()
        self.masks.clear()
