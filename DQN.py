import torch
import torch.optim as optim
import torch.nn.functional as F
from model import SimpleModel
from cartpole import CartPoleEnv
from collections import namedtuple, deque
import random
import math
from itertools import count

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN_Agent:
    def __init__(self, input_size, output_size, learning_rate=0.001):
        self.model = SimpleModel(input_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(10000)
        self.env = CartPoleEnv()

    def select_action(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                return self.model(torch.tensor(state, dtype=torch.float32)).max(0)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(2)]], dtype=torch.long)

    def optimize_model(self, batch_size, gamma=0.9):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch.next_state if s is not None])

        state_batch = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch.state])

        action_batch = torch.tensor([a for a in batch.action], dtype=torch.long).view(-1, 1)

        reward_batch = torch.cat([torch.tensor([r], dtype=torch.float32) for r in batch.reward])

        state_action_values = self.model(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size)
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * gamma) + reward_batch

        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, num_episodes, batch_size=128, gamma=0.999, epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=200):
        for i_episode in range(num_episodes):
            state = self.env.reset()
            for t in count():
                epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * i_episode / epsilon_decay)
                action = self.select_action(state, epsilon)
                next_state, reward, done, _ = self.env.step(action.item())
                if done:
                    next_state = None
                self.memory.push(state, action, next_state, reward)
                state = next_state
                self.optimize_model(batch_size, gamma)
                if done:
                    break

    def test(self, num_episodes=10):
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
