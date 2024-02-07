import torch
import random


class UCB:
    def __init__(self, **kwargs):
        """
        principle:
        UCB(s, a) = Q(s, a) + c * sqrt(log(n) / N(s, a))
        """
        self.c = kwargs['c']
        self.n = torch.tensor(kwargs['n'])

    def __call__(self, model, state, i_episode):
        with torch.no_grad():
            q_values = model(torch.tensor(state, dtype=torch.float32))
            exploration_bonus = self.c * torch.sqrt(torch.log(self.n) / q_values)
            return torch.argmax(q_values + exploration_bonus).view(1, 1)
