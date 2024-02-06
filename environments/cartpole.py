import gym

class CartPoleEnv:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.state = self.env.reset()

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return state, reward, done, info

    def reset(self):
        self.state = self.env.reset()
        return self.state
    
    @property
    def action_space(self):
        return self.env.action_space.n
    
    @property
    def observation_space(self):
        return self.env.observation_space.shape[0]
