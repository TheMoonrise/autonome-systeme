"""Function definitions for computing ppo"""

from .ppo_parameters import Parameters


def ppo_returns(params: Parameters, rewards, masks, values, value_next):
    """
    Computes the returns for the given rewards and values.
    :param params: The hyperparameters for the algorithm.
    :param rewards: The rewards received in each time step.
    :param masks: A mask filtering terminal states.
    :param values: The values estimated for each state.
    :param next_value: The value estimated for the upcoming state.
    """
    values = values + [value_next]
    advantage = 0
    returns = []

    for t in reversed(range(len(rewards))):
        d = rewards[t] + params.gamma * values[t + 1] * masks[t] - values[t]
        advantage = d + params.gamma * params.lmbda * masks[t] * advantage
        returns.insert(0, advantage + values[t])

    return returns
