import torch
import torch.nn.functional as F
from itertools import count
from collections import namedtuple
from utils import ReplayMemory
from algorithms import Base_Agent


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ActorCritic_Agent(Base_Agent):
    def __init__(self, model, optimizer, env, **kwargs):
        super().__init__(model, optimizer, env, **kwargs)
        self.actor_model = model(**kwargs['model_args'], AC='actor')
        self.critic_model = model(**kwargs['model_args'], AC='critic')
        self.actor_optimizer = optimizer(self.actor_model.parameters(), **kwargs['optimizer_args'])
        self.critic_optimizer = optimizer(self.critic_model.parameters(), **kwargs['optimizer_args'])
        self.memory = ReplayMemory(capacity=10000, Transition=Transition)
        self.env = env

    def optimize_model(self, batch_size, gamma=0.999):
        """
            principle:
            
        """
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch.next_state if s is not None])

        state_batch = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch.state])
        action_batch = torch.tensor([a for a in batch.action], dtype=torch.long).view(-1, 1)
        reward_batch = torch.cat([torch.tensor([r], dtype=torch.float32) for r in batch.reward])

        values = self.critic_model(state_batch).squeeze()
        next_values = torch.zeros(batch_size)
        next_values[non_final_mask] = self.critic_model(non_final_next_states).squeeze().detach()
        expected_values = (next_values * gamma) + reward_batch
        critic_loss = F.mse_loss(values, expected_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        state_action_values = self.actor_model(state_batch)
        action_log_probs = F.log_softmax(state_action_values, dim=-1)
        action_log_probs = action_log_probs.gather(1, action_batch).squeeze()
        actor_loss = (-action_log_probs * expected_values).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def train(self, num_episodes, batch_size=128, gamma=0.999, **kwargs):
        for i_episode in range(num_episodes):
            state = self.env.reset()
            for t in count():
                action = self.select_action(state, i_episode)
                next_state, reward, done, _ = self.env.step(action.item())
                if done:
                    next_state = None
                self.memory.push(state, action, next_state, reward)
                state = next_state
                self.optimize_model(batch_size, gamma)
                if done:
                    break

    @property
    def model(self):
        return self.actor_model
