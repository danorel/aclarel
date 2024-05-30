import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import pathlib
import random
import numpy as np
from collections import deque, namedtuple
import environments.cart_pole.environment as cart_pole_env
import environments.cart_pole.rl_methods as cart_pole_rl

agent_name = pathlib.Path(__file__).resolve().stem

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

def add_gradient_logging(model, threshold=1e-6):
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            def hook_function(grad, name=name):
                grad_norm = grad.norm().item()
                if grad_norm < threshold:
                    print(f"Vanishing grad detected for {name}: {grad_norm}")

            parameter.register_hook(hook_function)

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=24):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent(cart_pole_rl.Agent):
    def __init__(self, curriculum_name, use_pretrained: bool = False):
        super().__init__(agent_name, curriculum_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        self.autocast = False
        print(f"Autocast gradients: {self.autocast}")
        self.hyperparameters = {
            "total_episodes": 2000,
            "alpha": 0.0015,
            "gamma": 0.99,
            "replay_buffer_size": 50000,
            "batch_size": 128,
            "initial_epsilon": 1.0,
            "minimum_epsilon": 0.01,
            "epsilon_decay": 0.999,
            "print_interval": 10,
            "evaluation_interval": 10,
            "update_interval": 1000
        }
        self.hyperparameter_path = f"alpha-{self.hyperparameters['alpha']}_gamma-{self.hyperparameters['gamma']}_episodes-{self.hyperparameters['total_episodes']}"
        self.current_model: DQNNetwork = None
        self.target_model: DQNNetwork = None
        if use_pretrained:
            self.deserialize_agent()
        else:
            self.refresh_agent()
        self.replay_buffer = deque(maxlen=self.hyperparameters['replay_buffer_size'])
        self.optimizer = optim.Adam(self.current_model.parameters(), lr=self.hyperparameters['alpha'])
        self.epsilon = self.hyperparameters['initial_epsilon']
        self.steps_count = 0
    
    def act(self, state, greedily: bool = False):
        is_exploratory = False

        if greedily:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_values = self.current_model(state_tensor)
                action = torch.argmax(action_values).item()
        else:
            epsilon_decay = self.hyperparameters['epsilon_decay']
            minimum_epsilon = self.hyperparameters['minimum_epsilon']

            is_exploratory = False
            if np.random.random() < self.epsilon:
                is_exploratory = True
                action = cart_pole_env.env.action_space.sample()
            else:
                is_exploratory = False
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    action_values = self.current_model(state_tensor)
                    action = torch.argmax(action_values).item()

            self.epsilon *= epsilon_decay
            self.epsilon = max(minimum_epsilon, self.epsilon)
        
        return action, is_exploratory

    def train(self, prev_state, action, reward, next_state, done):
        self.replay_buffer.append(Transition(prev_state, action, reward, next_state, done))
        if len(self.replay_buffer) < self.hyperparameters['batch_size']:
            return

        transitions = random.sample(self.replay_buffer, self.hyperparameters['batch_size'])
        states, actions, rewards, next_states, dones = zip(*transitions)

        state_batch = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_state_batch = torch.tensor([s for s, done in zip(next_states, dones) if not done], dtype=torch.float32, device=self.device)

        action_batch = torch.tensor(actions, dtype=torch.long, device=self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        non_final_mask = torch.tensor([not done for done in dones], dtype=torch.bool, device=self.device)
        state_action_values = self.current_model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(-1)
        next_state_values = torch.zeros(self.hyperparameters['batch_size'], device=self.device)

        if non_final_mask.any():
            next_values = self.target_model(next_state_batch).max(1)[0].detach()
            next_state_values[non_final_mask] = next_values

        next_state_values = torch.zeros(self.hyperparameters['batch_size'], device=self.device)
        next_state_values[non_final_mask] = self.target_model(next_state_batch).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.hyperparameters['gamma']) + reward_batch

        if self.autocast:
            with autocast():
                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
                self.writer.add_scalar('Loss/loss', loss.item(), self.steps_count)
                scaler = GradScaler()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
        else:
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
            self.writer.add_scalar('Loss/loss', loss.item(), self.steps_count)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.steps_count % self.hyperparameters['update_interval'] == 0:
            self.target_model.load_state_dict(self.current_model.state_dict())

        self.writer.add_scalar('Performance/Reward', torch.mean(reward_batch).item(), self.steps_count)
        self.steps_count += 1

    def refresh_agent(self):
        self.current_model = DQNNetwork(cart_pole_env.state_size, cart_pole_env.action_size).to(self.device)
        self.target_model = DQNNetwork(cart_pole_env.state_size, cart_pole_env.action_size).to(self.device)
        self.target_model.load_state_dict(self.current_model.state_dict())
        add_gradient_logging(self.current_model)

    def serialize_agent(self):
        model_path = self.model_dir / f'{self.hyperparameter_path}.pth'
        torch.save(self.current_model.state_dict(), model_path)

    def deserialize_agent(self):
        model_path = self.model_dir / f'{self.hyperparameter_path}.pth'
        self.current_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.target_model.load_state_dict(self.current_model.state_dict())