import inspect
import torch
import random
import math


class EpsilonGreedy:
    def __init__(self, **kwargs):
        """
        principle:
        ε-greedy = argmax(Q(s, a)) with probability 1 - ε
                   random action with probability ε
        """
        self.epsilon_start = kwargs['epsilon_start']
        self.epsilon_end = kwargs['epsilon_end']
        self.epsilon_decay = kwargs['epsilon_decay']
        self.n = kwargs['n']

    def __call__(self, model, state, i_episode):
        if inspect.stack()[2].function == 'train':
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * i_episode / self.epsilon_decay)
            if random.random() > epsilon:
                with torch.no_grad():
                    return model(torch.tensor(state, dtype=torch.float32)).max(0)[1].view(1, 1)
            else:
                return torch.tensor([[random.randrange(self.n)]], dtype=torch.long)
        else:
            with torch.no_grad():
                return model(torch.tensor(state, dtype=torch.float32)).max(0)[1].view(1, 1)
    