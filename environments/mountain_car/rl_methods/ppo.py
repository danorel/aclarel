import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pathlib
import random
from torch.cuda.amp import autocast, GradScaler
from collections import deque, namedtuple
import environments.mountain_car.environment as mountain_car_env
import environments.mountain_car.rl_methods as mountain_car_rl

agent_name = pathlib.Path(__file__).resolve().stem

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'log_prob'))

def add_gradient_logging(model, threshold=1e-6):
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            def hook_function(grad, name=name):
                grad_norm = grad.norm().item()
                if grad_norm < threshold:
                    print(f"Vanishing grad detected for {name}: {grad_norm}")

            parameter.register_hook(hook_function)

class PPONetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=32):
        super(PPONetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size)
        self.actor_fc = nn.Linear(hidden_size, action_size)
        self.critic_fc = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action_probs = F.softmax(self.actor_fc(x), dim=-1)
        values = self.critic_fc(x).squeeze(-1)
        return action_probs, values

class PPOAgent(mountain_car_rl.Agent):
    def __init__(self, curriculum_name, use_pretrained: bool = False):
        super().__init__(agent_name, curriculum_name)
        self.device = device
        print(f"Device: {self.device}")
        self.autocast = device == 'cuda'
        self.scaler = GradScaler()
        print(f"Autocast gradients: {self.autocast}")
        self.hyperparameters = {
            "total_episodes": 250,
            "alpha": 0.001,
            "gamma": 0.99,
            "replay_buffer_size": 2000,
            "batch_size": 128,
            "clip_epsilon": 0.2,
            "gae_lambda": 0.99,
            "print_interval": 5,
            "evaluation_interval": 10,
            "train_interval": 5,
            'log_interval': 10,
        }
        self.hyperparameter_path = f"alpha-{self.hyperparameters['alpha']}_gamma-{self.hyperparameters['gamma']}_episodes-{self.hyperparameters['total_episodes']}"
        self.model = PPONetwork(mountain_car_env.env.observation_space.shape[0], mountain_car_env.env.action_space.n).to(self.device)
        if use_pretrained:
            self.deserialize_agent()
        else:
            self.refresh_agent()
        self.replay_buffer = deque(maxlen=self.hyperparameters['replay_buffer_size'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hyperparameters['alpha'])
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=1e-4)

    def preprocess_single(self, state):
        return torch.tensor(state, dtype=torch.float32).to(self.device)
    
    def preprocess_batch(self, states):
        state_tensors = torch.stack([torch.tensor(state, dtype=torch.float32).to(self.device) for state in states])
        return state_tensors.to(self.device)
    
    def act(self, state, greedily: bool = False):
        state_tensor = self.preprocess_single(state).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.model(state_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), dict(is_exploratory=False, log_prob=log_prob) 
    
    def compute_gae(self, next_values, rewards, dones, values):
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        gae = 0
        dones = dones.float()
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.hyperparameters['gamma'] * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.hyperparameters['gamma'] * self.hyperparameters['gae_lambda'] * gae * (1 - dones[t])
            advantages[t] = gae
            returns[t] = gae + values[t]
        return returns, advantages

    def compute_loss(self, action_probs, values, actions, rewards, next_states, dones, log_probs):
        values = values.squeeze(-1)
        
        returns, advantages = self.compute_gae(next_states, rewards, dones, values)

        probs_distribution = torch.distributions.Categorical(action_probs)

        old_log_probs = log_probs.clone()
        new_log_probs = probs_distribution.log_prob(actions).squeeze()

        ratios = torch.exp(new_log_probs - old_log_probs)

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.hyperparameters['clip_epsilon'], 1.0 + self.hyperparameters['clip_epsilon']) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(values, returns)

        total_loss = policy_loss + 0.5 * value_loss

        return total_loss

    def train(self, prev_state, action, reward, next_state, done, log_prob):
        self.steps_count += 1

        self.replay_buffer.append(Transition(prev_state, action, reward, next_state, done, log_prob))
        
        if len(self.replay_buffer) < self.hyperparameters['batch_size']:
            return
        
        if self.steps_count % self.hyperparameters['train_interval'] != 0:
            return

        transitions = random.sample(self.replay_buffer, self.hyperparameters['batch_size'])
        states, actions, rewards, next_states, dones, log_probs = zip(*transitions)

        state_batch = self.preprocess_batch(states)
        next_state_batch = self.preprocess_batch(next_states)
        action_batch = torch.tensor(actions, dtype=torch.long, device=self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(dones, dtype=torch.bool, device=self.device)
        log_prob_batch = torch.tensor(log_probs, dtype=torch.float32, device=self.device)

        action_probs, state_values = self.model(state_batch)
        _, next_state_values = self.model(next_state_batch)
        
        next_state_values = next_state_values.detach()

        total_loss = self.compute_loss(action_probs, state_values, action_batch, reward_batch, next_state_values, done_batch, log_prob_batch)

        self.optimizer.zero_grad()
        if self.autocast:
            with autocast():
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        self.lr_scheduler.step() 

        if self.steps_count % self.hyperparameters['log_interval'] == 0:
            self.writer.add_scalar('Loss/loss', total_loss.item(), self.steps_count)

    def refresh_agent(self):
        add_gradient_logging(self.model)

    def serialize_agent(self):
        model_path = self.model_dir / f'{self.hyperparameter_path}.pth'
        torch.save(self.model.state_dict(), model_path)

    def deserialize_agent(self):
        model_path = self.model_dir / f'{self.hyperparameter_path}.pth'
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
