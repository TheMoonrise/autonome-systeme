"""Handles training and evaluation of ppo networks"""

import torch

from gym import Env
from torch.optim import Optimizer

from .ppo_parameters import Parameters
from .ppo_actor_critic import ActorCritic
from .ppo_update import Update
from .ppo_functions import ppo_returns


def ppo_train(env: Env, params: Parameters, model: ActorCritic, optimizer: Optimizer, device: str, f_evaluate=None):
    """
    Performs a ppo training loop on the given model and environment.
    :param env: The open ai gym environment to train on.
    :param params: The parameters used for the training.
    :param model: The ppo actor critic model trained on the environment.
    :param optimizer: The optimizer configured for the model and used for improving based on collected experience.
    :param device: String property naming the device used for training.
    :param f_evaluate: Function called to evaluate the current model performance.
    """
    print("Begin ppo training")
    state = env.reset()

    for i in range(params.training_iterations):
        print(f"\rTraining iteration {i:0>4}", end='')

        trace = _trace(env, params, model, device, state)
        state, states, actions, probs, values, rewards, masks = trace

        # compute the discounted returns for each state in the trace
        state_next = torch.FloatTensor(state).unsqueeze(0).to(device)
        _, value_next = model(state_next)
        returns = ppo_returns(params, rewards, masks, values, value_next.squeeze(1))

        # combine the trace in tensors and update the model
        states = torch.stack(states)
        actions = torch.stack(actions)
        probs = torch.stack(probs).detach()
        values = torch.stack(values).detach()

        returns = torch.stack(returns).detach()
        advantages = returns - values

        update = Update(params, states, actions, probs, returns, advantages)
        update.update(optimizer, model)

        # at certain intervals measure the model performance
        if i % min(params.training_iterations // 10, 250) == 0 or i == (params.training_iterations - 1):
            print()
            if not f_evaluate: continue
            f_evaluate()


def ppo_evaluate(env: Env, model: ActorCritic, device: str, count: int, render: bool):
    """
    Runs an complete episode on the given environment for evaluation.
    :param env: The open ai gym environment to evaluate on.
    :param model: The ppo actor critic model to be evaluated.
    :param device: String property naming the device used for the model.
    :param count: The number of iteratios run on the environment.
    :param render: Whether the environment should be rendered.
    :returns: The total reward collected over the evaluation.
    """
    reward_mean = 0

    for i in range(count):
        state = env.reset()
        reward_total = 0
        done = False

        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            dist, _ = model(state)
            action = dist.sample().cpu().numpy()

            state_next, reward, done, _ = env.step(action)
            state = state_next.squeeze()

            reward_total += reward[0]
            if render and i == 0: env.render()

        reward_mean += reward_total
    return reward_mean / count


def _trace(env: Env, params: Parameters, model: ActorCritic, device: str, state):
    """
    Samples a trace of training data from the environment
    :param env: The open ai gym environment to train on.
    :param params: The parameters used for the training.
    :param model: The ppo actor critic model trained on the environment.
    :param device: String property naming the device used for training.
    :param state: The initial state the environment is currently in
    :returns: The state the environment is currently in.
    :returns: The list of collected state values.
    :returns: The list of collected actions.
    :returns: The list of action probabilities for each action in the action list.
    :returns: The critic state value for each state in the state list.
    :returns: The list of rewards received in each step during the trace.
    :returns: The list of masks for filtering terminal states where 0 indicates a terminal state.
    """
    # define lists to hold the collected training data
    states, actions, probs, values = [], [], [], []
    rewards, masks, = [], []

    done = [False]

    # collect trace data for updating the model
    for _ in range(params.trace):
        if all(done): state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        states.append(state)

        dist, value = model(state)
        values.append(value.squeeze(1))

        action = dist.sample()
        actions.append(action)

        state_next, reward, done, _ = env.step(action.cpu().numpy())
        state = state_next.squeeze()
        if isinstance(done, bool): done = [done]

        rewards.append(torch.FloatTensor(reward).to(device))
        masks.append(torch.FloatTensor(done).mul(-1).add(1).to(device))

        prob = dist.log_prob(action)
        probs.append(prob)

    return state, states, actions, probs, values, rewards, masks
