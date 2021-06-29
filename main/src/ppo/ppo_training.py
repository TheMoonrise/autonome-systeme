"""Handles training and evaluation of ppo networks"""

import mlflow
import torch
import numpy as np

from gym import Env
from torch.optim import Optimizer
from datetime import datetime

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
        self.episode_steps = 0

    def train(self, params: Parameters, optimizer: Optimizer, device: str, save_interval: int = -1):
        """
        Performs a ppo training loop on the given model and environment.
        :param params: The parameters used for the training.
        :param optimizer: The optimizer configured for the model and used for improving based on collected experience.
        :param device: String property naming the device used for training.
        :param save_interval: The interval at which to save the current model.
        """
        self.state = self.env.reset()
        self.done = [False]

        self.performance.clear()
        self.performance_counter = 0
        self.start_time = datetime.now()

        print(f'\nBegin Training [{self.model.id}]')
        print('Iteration, Epoch, Performance, ETA')

        for i in range(1, params.training_iterations + 1):
            performance = np.average(self.performance[-10:]) if self.performance else 0
            eta = str((datetime.now() - self.start_time) * ((params.training_iterations - i) / i))
            if '.' in eta: eta = eta[:eta.rindex('.')]

            print(f'\r[ {i:^5} | {len(self.performance):^5} | {performance:^5.0f} | {eta} ]', end='  ')

            self._clear_trace()
            self._collect_trace(params, device)

            # compute the discounted returns for each state in the trace
            state_next = self._shaped_state_tensor(self.state, device)
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
            update.update(optimizer, self.model, i)

            if save_interval > 0 and (i % save_interval == 0 or i == params.training_iterations):
                appendix = f'-{i:0>4}-{performance:.0f}'
                self.model.save(appendix, optimizer)

                if params.mlflow:
                    mlflow.log_artifact(self.model.model_path(appendix, is_save=True))
                    mlflow.log_artifact(self.model.optimizer_path(appendix, is_save=True))

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

        while not all(self.done):
            self.state = self._shaped_state_tensor(self.state)
            dist, _ = self.model(self.state)
            action = dist.sample().cpu().numpy()

            state_next, reward, self.done, _ = self.env.step(action)
            self.state = state_next.squeeze()

            if isinstance(self.done, bool): self.done = [self.done]

            self.performance_counter += np.mean(reward)
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
            self.state = self._shaped_state_tensor(self.state, device)
            self.states.append(self.state)

            dist, value = self.model(self.state)
            self.values.append(value.squeeze(1))

            action = dist.sample()
            self.actions.append(action)

            state_next, rewards, self.done, _ = self.env.step(action.cpu().numpy())
            self.state = state_next.squeeze()

            # pendulum environment provides only a single bool
            if isinstance(self.done, bool): self.done = [self.done]

            self.rewards.append(torch.tensor(rewards, device=device, dtype=torch.float32))
            self.masks.append(torch.tensor(self.done, device=device, dtype=torch.float32).mul(-1).add(1))

            prob = dist.log_prob(action)
            self.probs.append(prob)

            # update performance tracking for the first agent
            self.performance_counter += rewards[0]
            self.episode_steps += 1

            if self.done[0]:
                if params.mlflow:
                    mlflow.log_metric('performance', self.performance_counter, step=len(self.performance))
                    mlflow.log_metric('episode length', self.episode_steps, step=len(self.performance))

                self.performance.append(self.performance_counter)
                self.performance_counter = 0
                self.episode_steps = 0

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

    def _shaped_state_tensor(self, state: np.ndarray, device: str = 'cpu'):
        """
        Converts the given state to a well-formed torch tensor.
        :param state: The state numpy array to modify.
        :param device: The device to move the tensor to.
        :returns: A tensor of the shape (num_agents, state_size)
        """
        state = torch.tensor(state, device=device, dtype=torch.float32)
        # some (pendulum) environment states are only one dimensional
        if len(state.shape) < 2: state = state.unsqueeze(0)
        return state
