import torch
import numpy as np

def Boltzmann(model, state, temperature):
    """
        principle:
        action probabilities follow a softmax distribution over Q values, controlled by temperature τ
    """
    with torch.no_grad():
        q_values = model(torch.tensor(state, dtype=torch.float32))
        probabilities = torch.nn.functional.softmax(q_values / temperature, dim=0)
        action = torch.multinomial(probabilities, 1)
    return action
