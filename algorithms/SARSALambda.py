import torch
import torch.nn.functional as F
import math
from itertools import count
from collections import namedtuple
from utils import ReplayMemory
from algorithms import Base_Agent


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'next_action', 'reward'))

class SARSALambda_Agent(Base_Agent):
    def __init__(self, model, optimizer, env, lambda_=0.9, **kwargs):
        super().__init__(model, optimizer, env, lambda_=0.9, **kwargs)
        self.model = model(**kwargs['model_args'])
        self.optimizer = optimizer(self.model.parameters(), **kwargs['optimizer_args'])
        self.memory = ReplayMemory(capacity=10000, Transition=Transition)
        self.env = env
        self.lambda_ = lambda_
        self.eligibility_trace = {name: torch.zeros_like(param, memory_format=torch.preserve_format) 
                                  for name, param in self.model.named_parameters()}

    def optimize_model(self, batch_size, gamma=0.999):
        """
            principle:
            Q'(s, a) = Q(s, a) + α * E(s, a) * [r + γ * Q(s', a') - Q(s, a)]
                     = [1 - α * E(s, a)] * Q(s, a) + [α * E(s, a)] * [r + γ * Q(s', a')]
            E'(s, a) = γ * λ * E(s, a) + 1
        """
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch.next_state if s is not None])
        non_final_next_actions = torch.stack([a for a in batch.next_action if a is not None])

        state_batch = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch.state])
        action_batch = torch.tensor([a for a in batch.action], dtype=torch.long).view(-1, 1)
        reward_batch = torch.cat([torch.tensor([r], dtype=torch.float32) for r in batch.reward])

        state_action_values = self.model(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size)
        next_state_values[non_final_mask] = self.model(non_final_next_states).gather(1, non_final_next_actions.squeeze().unsqueeze(-1)).squeeze().detach()

        expected_state_action_values = (next_state_values * gamma) + reward_batch


        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            for param in self.model.parameters():
                self.eligibility_trace[param] = gamma * self.lambda_ * self.eligibility_trace[param] + param.grad
                param.grad = self.eligibility_trace[param]

        self.optimizer.step()

    def train(self, num_episodes, batch_size=128, gamma=0.999, epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=200, n=2, **kwargs):
        for i_episode in range(num_episodes):
            for param in self.model.parameters():
                self.eligibility_trace[param] = torch.zeros_like(param)

            state = self.env.reset()
            action = self.select_action(state, epsilon_start, n)
            for t in count():
                epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * i_episode / epsilon_decay)
                next_state, reward, done, _ = self.env.step(action.item())
                next_action = self.select_action(next_state, epsilon, n) if not done else None
                if done:
                    next_state = None
                self.memory.push(state, action, next_state, next_action, reward)
                state, action = next_state, next_action
                self.optimize_model(batch_size, gamma)
                if done:
                    break
