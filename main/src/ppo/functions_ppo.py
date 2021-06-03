"""Function definitions for computing ppo"""


def ppo_returns(params, rewards, masks, values, next_values):
    """
    Computes the returns for the given rewards and values.
    :param params: The hyperparameters for the algorithm.
    :param rewards: The rewards received in each time step.
    :param masks: A mask filtering terminal states.
    :param values: The values estimated for each state.
    :param next_value: The value estimated for the upcoming state.
    """
    assert 'gamma' in params
    assert 'tau' in params

    gamma = params['gamma']
    tau = params['tau']

    values = values + [next_values]
    advantage = 0
    returns = []

    for t in reversed(range(len(rewards))):
        d = rewards[t] + gamma * values[t + 1] * masks[t] - values[t]
        advantage = d + gamma * tau * masks[t] * advantage
        returns.insert(0, advantage + values[t])

    return returns
