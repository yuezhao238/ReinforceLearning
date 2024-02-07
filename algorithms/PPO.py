import torch
import torch.nn.functional as F
from itertools import count
from collections import namedtuple
from utils import ReplayMemory
from algorithms import Base_Agent
from torch.distributions import Categorical

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'log_prob'))

class PPO_Agent(Base_Agent):
    def __init__(self, model, optimizer, env, **kwargs):
        super().__init__(model, optimizer, env, **kwargs)
        self.actor_model = model(**kwargs['model_args'], AC='actor')
        self.critic_model = model(**kwargs['model_args'], AC='critic')
        self.actor_optimizer = optimizer(self.actor_model.parameters(), **kwargs['optimizer_args'])
        self.critic_optimizer = optimizer(self.critic_model.parameters(), **kwargs['optimizer_args'])
        self.memory = ReplayMemory(capacity=10000, Transition=Transition)
        self.env = env
        self.eps_clip = kwargs['train_args']['eps_clip']
        self.gae_lambda = kwargs['train_args']['gae_lambda']
        self.entropy_beta = kwargs['train_args']['entropy_beta']
    
    def compute_gae_and_returns(self, rewards, values, gamma=0.999):
        gae = 0
        returns = []
        advantages = torch.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + (gamma * values[t + 1] if t + 1 < len(rewards) else 0) - values[t]
            gae = delta + gamma * self.gae_lambda * gae
            advantages[t] = gae
            returns.insert(0, gae + values[t])
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        return advantages, returns

    def optimize_model(self, batch_size, gamma=0.999):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch.state])
        action_batch = torch.tensor([a for a in batch.action], dtype=torch.long).view(-1, 1)
        reward_batch = torch.cat([torch.tensor([r], dtype=torch.float32) for r in batch.reward])
        log_prob_batch = torch.stack(batch.log_prob)

        values = self.critic_model(state_batch).squeeze()
        advantages, returns = self.compute_gae_and_returns(reward_batch, values)

        action_probs = self.actor_model(state_batch)
        dist = Categorical(logits=action_probs)
        new_log_probs = dist.log_prob(action_batch.squeeze(-1))
        ratio = (new_log_probs - log_prob_batch).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_beta * dist.entropy().mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        critic_loss = F.mse_loss(values, returns)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def train(self, num_episodes, batch_size=128, gamma=0.999, **kwargs):
        for i_episode in range(num_episodes):
            state = self.env.reset()
            for t in count():
                action, log_prob = self.select_action(state, i_episode)
                next_state, reward, done, _ = self.env.step(action.item())
                if done:
                    next_state = None
                self.memory.push(state, action, next_state, reward, log_prob)
                state = next_state
                self.optimize_model(batch_size, gamma)
                if done:
                    break

    @property
    def model(self):
        return self.actor_model
