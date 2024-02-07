import torch


class Boltzmann:
    def __init__(self, **kwargs):
        """
        principle:
        action probabilities follow a softmax distribution over Q values, controlled by temperature τ
        """
        self.temperature = kwargs['temperature']

    def __call__(self, model, state, i_episode):
        with torch.no_grad():
            q_values = model(torch.tensor(state, dtype=torch.float32))
            probabilities = torch.nn.functional.softmax(q_values / self.temperature, dim=0)
            action = torch.multinomial(probabilities, 1)
        return action, torch.log(probabilities[action].view(1, 1))
