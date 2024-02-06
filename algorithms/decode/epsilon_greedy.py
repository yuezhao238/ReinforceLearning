import torch
import random


def EpsilonGreedy(model, state, epsilon, n):
    """
        principle:
        ε-greedy = argmax(Q(s, a)) with probability 1 - ε
                   random action with probability ε
    """
    if random.random() > epsilon:
        with torch.no_grad():
            return model(torch.tensor(state, dtype=torch.float32)).max(0)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n)]], dtype=torch.long)
    