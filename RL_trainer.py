from collections import OrderedDict
from cartpole import CartPoleEnv
from DQN import DQN_Agent
from model import SimpleModel
import torch.optim as optim

class RLTrainer:
    def __init__(self, env, algorithm, config):
        self.env = env
        self.algorithm = algorithm
        self.config = config

    def train(self):
        self.algorithm.train(**self.config['train_args'])

    def test(self):
        self.algorithm.test(**self.config['test_args'])

    def run(self):
        self.test()
        self.train()
        self.test()

def main():
    config = OrderedDict(
        algorithm_args = OrderedDict(
            train_args = OrderedDict(
                num_episodes=200,
                batch_size=128,
                gamma=0.999,
                epsilon_start=0.9,
                epsilon_end=0.05,
                epsilon_decay=200
            ),
            test_args = OrderedDict(
                num_episodes=10
            )
        ),
        model_args = OrderedDict(
            input_size=4,
            output_size=2
        ),
        optimizer_args = OrderedDict(
            lr=0.001
        )
    )

    env = CartPoleEnv()
    model = SimpleModel(**config['model_args'])
    optimizer = optim.Adam(model.parameters(), lr=config['optimizer_args']['lr'])
    algorithm = DQN_Agent(model, optimizer, env)
    trainer = RLTrainer(env, algorithm, config['algorithm_args'])

    trainer.run()

main()