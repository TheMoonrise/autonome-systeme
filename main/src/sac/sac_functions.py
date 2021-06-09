import random

import gym
import numpy as np

"""
Fixed-size buffer to store experience tuples.
"""
class ReplayBuffer:
    """
    Initializes the the ReplayBuffer object.
    :param capacity: The capacity of the buffer.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    """
    Adds a new experience to memory.
    :param state:
    :param action:
    :param reward:
    :param next_state:
    :param done:
    """
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    """
    Randomly sample a batch of experiences from memory.
    :param batch_size: Maximum size of buffer.
    """
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    """
    Returns length of buffer.
    """
    def __len__(self):
        return len(self.buffer)

class NormalizedActions(gym.ActionWrapper):
    """
    """
    def action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action
    """
    """
    def reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return action