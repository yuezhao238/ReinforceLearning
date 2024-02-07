from itertools import count
from algorithms.decode import (
    EpsilonGreedy,
    UCB,
    Boltzmann,
)


class Base_Agent:
    def __init__(self, model, optimizer, env, **kwargs):
        self.config = kwargs
        self.strategy = self.get_strategy()

    def select_action(self, state, i_episode):
        return self.strategy(self.model, state, i_episode)

    def optimize_model(self, batch_size, gamma=0.999):
        raise NotImplementedError

    def train(self, num_episodes, batch_size=128, gamma=0.999, **kwargs):
        raise NotImplementedError

    def test(self, num_episodes=10, **kwargs):
        for i_episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            for t in count():
                action = self.select_action(state, i_episode)
                next_state, reward, done, _ = self.env.step(action.item())
                total_reward += reward
                state = next_state
                if done:
                    print(f"Test Episode {i_episode} finished after {t+1} timesteps with total reward {total_reward}.")
                    break

    def get_strategy(self):
        return globals()[self.config['decode_args']['strategy']](**self.config['decode_args'])
