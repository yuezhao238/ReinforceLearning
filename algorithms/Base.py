from itertools import count
from collections import namedtuple


class Base_Agent:
    def __init__(self, model, optimizer, env, **kwargs):
        pass

    def select_action(self, state, epsilon):
        raise NotImplementedError

    def optimize_model(self, batch_size, gamma=0.999):
        raise NotImplementedError

    def train(self, num_episodes, batch_size=128, gamma=0.999, epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=200, **kwargs):
        raise NotImplementedError

    def test(self, num_episodes=10, **kwargs):
        for i_episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            for t in count():
                action = self.select_action(state, epsilon=0)
                next_state, reward, done, _ = self.env.step(action.item())
                total_reward += reward
                state = next_state
                if done:
                    print(f"Test Episode {i_episode} finished after {t+1} timesteps with total reward {total_reward}.")
                    break
