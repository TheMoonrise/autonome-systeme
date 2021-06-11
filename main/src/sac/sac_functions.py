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
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences from memory.
        :param batch_size: Maximum size of buffer.
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        """
        Returns length of buffer.
        """
        return len(self.buffer)

class NormalizedActions(gym.ActionWrapper):
    
    def action(self, action):
        """
        TODO: add comment
        """
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action
    
    def reverse_action(self, action):
        """
        TODO: add comment
        """
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return action