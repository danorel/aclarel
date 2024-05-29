import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pathlib
import random
import numpy as np
from collections import deque, namedtuple
import environments.cart_pole.environment as cart_pole_env
import environments.cart_pole.rl_methods as cart_pole_rl

agent_name = pathlib.Path(__file__).resolve().stem

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

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
        self.hyperparameters = {
            "total_episodes": 2000,
            "alpha": 0.001,
            "gamma": 0.99,
            "replay_buffer_size": 50000,
            "batch_size": 64,
            "initial_epsilon": 1.0,
            "minimum_epsilon": 0.01,
            "epsilon_decay": 0.999,
            "print_interval": 25,
            "evaluation_interval": 10,
            "update_interval": 1000
        }
        self.hyperparameter_path = f"alpha-{self.hyperparameters['alpha']}_gamma-{self.hyperparameters['gamma']}_episodes-{self.hyperparameters['total_episodes']}"
        # Define DQN network
        self.current_model: DQNNetwork = None
        self.target_model: DQNNetwork = None
        if use_pretrained:
            self.deserialize_agent()
        else:
            self.refresh_agent()
        # Define rest DQN architecture parts
        self.replay_buffer = deque(maxlen=self.hyperparameters['replay_buffer_size'])
        self.optimizer = optim.Adam(self.current_model.parameters(), lr=self.hyperparameters['alpha'])
        # Parameters
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
        batch = Transition(*zip(*transitions))

        # Convert to tensors and ensure they are the correct shape
        state_batch = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch.state]).to(self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long).to(self.device).unsqueeze(-1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool, device=self.device)
        non_final_next_states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch.next_state if s is not None]).to(self.device)

        # Compute Q(s_t, a)
        state_action_values = self.current_model(state_batch).gather(1, action_batch).squeeze(-1)

        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.hyperparameters['batch_size'], device=self.device)
        if non_final_mask.any():
            next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.hyperparameters['gamma']) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.current_model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.steps_count % self.hyperparameters['update_interval'] == 0:
            self.target_model.load_state_dict(self.current_model.state_dict())
        self.steps_count += 1

    def refresh_agent(self):
        self.current_model = DQNNetwork(cart_pole_env.state_size, cart_pole_env.action_size).to(self.device)
        self.target_model = DQNNetwork(cart_pole_env.state_size, cart_pole_env.action_size).to(self.device)
        self.target_model.load_state_dict(self.current_model.state_dict())

    def serialize_agent(self):
        model_path = self.model_dir / f'{self.hyperparameter_path}.pth'
        torch.save(self.current_model.state_dict(), model_path)

    def deserialize_agent(self):
        model_path = self.model_dir / f'{self.hyperparameter_path}.pth'
        self.current_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.target_model.load_state_dict(self.current_model.state_dict())