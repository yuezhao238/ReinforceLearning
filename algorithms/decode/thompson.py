import numpy as np
from scipy.stats import beta

def ThompsonSampling(successes, failures):
    """
        principle:
        sample from a Beta distribution for each action and choose the action with the highest sample
    """
    samples = [beta(a=1 + s, b=1 + f).rvs() for s, f in zip(successes, failures)]
    return np.argmax(samples)
