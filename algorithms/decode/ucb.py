import torch
import random


def UCB(model, state, c, n):
    """
        principle:
        UCB(s, a) = Q(s, a) + c * sqrt(log(n) / N(s, a))
    """
    with torch.no_grad():
        q_values = model(torch.tensor(state, dtype=torch.float32))
        exploration_bonus = c * torch.sqrt(torch.log(n) / q_values)
        return q_values + exploration_bonus
    