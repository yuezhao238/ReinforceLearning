from collections import OrderedDict
from environments import CartPoleEnv
from algorithms import (
    DQN_Agent,
    SARSA_Agent,
    SARSALambda_Agent,
    ActorCritic_Agent,
)
# from dev import (
#     A3C_Agent,
# )
from models import SimpleModel
import torch.optim as optim
import argparse


class RLTrainer:
    def __init__(self, model, optimizer, env, config):
        self.env = env
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.lr = config['optimizer_args']['lr']
        self.algorithm = self._get_algorithm(config['algorithm_name'])
    
    def _get_algorithm(self, algorithm_name):
        return globals()[algorithm_name + "_Agent"](self.model, self.optimizer, self.env, **self.config)

    def train(self):
        self.algorithm.train(**self.config['train_args'])

    def test(self):
        self.algorithm.test(**self.config['test_args'])

    def run(self):
        self.test()
        self.train()
        self.test()

def main(args):
    config = OrderedDict(
        algorithm_name = args.algorithm,
        train_args = OrderedDict(
            num_episodes=200,
            batch_size=128,
            gamma=0.999,
            epsilon_start=0.9,
            epsilon_end=0.05,
            epsilon_decay=200,
            lambda_=0.9,
        ),
        test_args = OrderedDict(
            num_episodes=10,
        ),
        optimizer_args = OrderedDict(
            lr=0.001,
            alpha=0.99,
        ),
        model_args = OrderedDict(
            input_size=4,
            output_size=2,
        ),
    )

    env = CartPoleEnv()
    model = SimpleModel
    optimizer = optim.RMSprop
    trainer = RLTrainer(
        model=model,
        optimizer=optimizer,
        env=env,
        config=config
    )

    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm")
    args = parser.parse_args()
    main(args=args)
