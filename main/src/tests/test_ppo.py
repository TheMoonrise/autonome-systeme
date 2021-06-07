"""Tests to validate the ppo implementation"""

import torch

from ..ppo.ppo_parameters import Parameters
from ..ppo.ppo_functions import ppo_returns


def test_ppo_returns():
    """
    Tests the computed returns for ppo.
    """
    params = Parameters(3, 1)
    params.gamma = 1
    params.tau = 1

    rewards = [torch.tensor([1, 1, 0]), torch.tensor([0, 0, 0]), torch.tensor([1, 1, 1])]
    masks = [torch.tensor([1, 0, 1]), torch.tensor([1, 1, 1]), torch.tensor([1, 0, 1])]
    values = [torch.tensor([1, 1, 1]), torch.tensor([1, 1, 1]), torch.tensor([1, 1, 1])]
    value_next = torch.tensor([1, 1, 1])

    targets = [torch.tensor([3, 1, 2]), torch.tensor([2, 1, 2]), torch.tensor([2, 1, 2])]
    returns = ppo_returns(params, rewards, masks, values, value_next)
    print('Returns without discount (gamma = 1):', returns)

    assert all([torch.equal(x, y) for x, y in zip(targets, returns)])

    # introduce a discount factor and rerun test
    params.gamma = 0.5

    targets = [torch.tensor([1.375, 1, 0.375]), torch.tensor([0.75, 0.5, 0.75]), torch.tensor([1.5, 1, 1.5])]
    returns = ppo_returns(params, rewards, masks, values, value_next)
    print('Returns with discount (gamma = 0.5):', returns)

    assert all([torch.equal(x, y) for x, y in zip(targets, returns)])
