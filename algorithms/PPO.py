import torch
import torch.nn.functional as F
from itertools import count
from collections import namedtuple
from utils import ReplayMemory
from algorithms import Base_Agent
from torch.distributions import Categorical


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'log_prob', 'mask'))

class PPO_Agent(Base_Agent):
    def __init__(self, model, optimizer, env, **kwargs):
        super().__init__(model, optimizer, env, **kwargs)
        self.policy_model = model(**kwargs['model_args'], AC='actor')
        self.value_model = model(**kwargs['model_args'], AC='critic')
        self.p_optimizer = optimizer(self.policy_model.parameters(), **kwargs['optimizer_args'])
        self.v_optimizer = optimizer(self.value_model.parameters(), **kwargs['optimizer_args'])
        self.tau = kwargs['train_args'].get('tau', 0.95)
        self.eps_clip = kwargs['train_args'].get('eps_clip', 0.2)
        self.env = env
        self.memory = ReplayMemory(capacity=10000, Transition=Transition)

    def compute_gae(self, rewards, masks, values, gamma=0.999, tau=0.95):
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def ppo_iter(self, mini_batch_size, states, actions, log_probs, returns, advantages):
        batch_size = states.size(0)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = torch.randperm(batch_size)[:mini_batch_size]
            yield states[rand_ids, :], actions[rand_ids], log_probs[rand_ids], returns[rand_ids], advantages[rand_ids]
    
    def optimize_model(self, batch_size, next_value, gamma=0.999, tau=0.95):
        transitions = self.memory.fetch()
        batch = Transition(*zip(*transitions))
        rewards =[torch.tensor([r], dtype=torch.float32) for r in batch.reward]
        masks = [torch.tensor([m], dtype=torch.float32) for m in batch.mask]
        values = self.value_model(torch.stack(batch.state)).squeeze().detach()
        states = torch.stack(batch.state)
        actions = torch.tensor([a for a in batch.action], dtype=torch.long).view(-1, 1)

        returns = self.compute_gae(rewards, masks, torch.cat([values, next_value.squeeze().unsqueeze(0)]), gamma, tau)
        returns = torch.cat(returns)
        log_probs = torch.cat(batch.log_prob)
        advantages = returns - values
        for _ in range(3):
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(32, states, actions, log_probs, returns, advantages):
                dist = self.policy_model(state)
                dist = Categorical(dist)
                new_log_probs = dist.log_prob(action)
                entropy = dist.entropy().mean()
                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantage
                policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
                self.p_optimizer.zero_grad()
                policy_loss.backward()
                self.p_optimizer.step()

                value_loss = (return_ - self.value_model(state)).pow(2).mean()
                self.v_optimizer.zero_grad()
                value_loss.backward()
                self.v_optimizer.step()
        self.memory.clear()

    def train(self, num_episodes, batch_size=128, gamma=0.999, tau=0.95, **kwargs):
        for i_episode in range(num_episodes):
            state = self.env.reset()
            for t in count():
                state = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    dist = self.policy_model(state)
                    dist = Categorical(dist)
                
                action = dist.sample()
                log_prob = dist.log_prob(action)
                next_state, reward, done, _ = self.env.step(action.item())
                self.memory.push(state, action, next_state, reward, log_prob, 1-done)
                state = next_state
                if done:
                    break

            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            with torch.no_grad():
                next_value = self.value_model(next_state)
            self.optimize_model(batch_size, next_value, gamma, tau)

    @property
    def model(self):
        return self.policy_model
