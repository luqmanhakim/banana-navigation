from collections import deque
import numpy as np
import random
import torch


class ReplayBuffer():
    def __init__(self, buffer_size=int(1e4), device="cpu"):
        self.memory = deque(maxlen=buffer_size)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        to_add = np.array([state, action, reward, next_state, done])
        self.memory.append(to_add)

    def sample(self, size=100):
        sample_experiences = random.sample(self.memory, size)
        states = torch.from_numpy(np.vstack([e[0] for e in sample_experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in sample_experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in sample_experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in sample_experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in sample_experiences])).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.memory)
