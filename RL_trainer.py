from collections import OrderedDict
from cartpole import CartPoleEnv
from DQN import DQN_Agent

class RLTrainer:
    def __init__(self, env, algorithm, config):
        self.env = env
        self.algorithm = algorithm
        self.config = config

    def train(self):
        self.algorithm.train(**self.config)

    def test(self, num_episodes=10):
        self.algorithm.test(num_episodes)

    def run(self):
        self.test(1)
        self.train()
        self.test(5)

def main():
    config = OrderedDict(
        num_episodes=200,
        batch_size=128,
        gamma=0.999,
        epsilon_start=0.9,
        epsilon_end=0.05,
        epsilon_decay=200
    )

    env = CartPoleEnv()
    algorithm = DQN_Agent(4, 2)
    trainer = RLTrainer(env, algorithm, config)

    trainer.run()

main()