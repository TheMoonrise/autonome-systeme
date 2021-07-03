import random

import gym
import numpy as np

"""
Fixed-size buffer to store experience tuples.
"""


class ReplayBuffer:
    def __init__(self, capacity):
        """
        Initializes the the ReplayBuffer object.
        :param capacity: The capacity of the buffer.
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """
        Adds a new experience to memory.
        :param state: current state
        :param action: chosen action
        :param reward: obtained reward
        :param next_state: new state after the action was executed
        :param done: flag that indicates if the episode has reached a terminal state
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences from memory.
        :param batch_size: Maximum size of the replay buffer.
        :returns: random sample from the buffer
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        """
        Returns length of the replay buffer.
        """
        return len(self.buffer)


class NormalizedActions(gym.ActionWrapper):

    def action(self, action):
        """
        gets the original action value corresponding to the normalized action value 
        :param action: normalized action value
        :returns: action value
        """
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action):
        """
        normalizes the action space onto the interval [-1, 1]
        :param action: action value
        :returns: normalized action value
        """
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action
