import numpy as np
from scipy.stats import beta


class ThompsonSampling:
    def __init__(self, **kwargs):
        raise NotImplementedError
        """
        principle:
        sample from a Beta distribution for each action and choose the action with the highest sample
        """
        self.successes = kwargs['successes']
        self.failures = kwargs['failures']
    
    def __call__(self, i_episode):
        raise NotImplementedError
        samples = [beta(a=1 + s, b=1 + f).rvs() for s, f in zip(self.successes, self.failures)]
        return np.argmax(samples)
