import torch
import torch.nn as nn
import torch.optim as optim
import pathlib
import random
import numpy as np
from collections import deque
import environments.cart_pole.environment as cart_pole_env
import environments.cart_pole.rl_methods as cart_pole_rl

agent_name = pathlib.Path(__file__).resolve().stem

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size = 24):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class DQNAgent(cart_pole_rl.Agent):
    def __init__(self, curriculum_name, use_pretrained: bool = False):
        super().__init__(agent_name, curriculum_name)
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
                state = torch.FloatTensor(state).unsqueeze(0)
                action_values = self.current_model(state)
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
                    state = torch.FloatTensor(state).unsqueeze(0)
                    action_values = self.current_model(state)
                    action = torch.argmax(action_values).item()

            self.epsilon *= epsilon_decay
            self.epsilon = max(minimum_epsilon, self.epsilon)
        
        return action, is_exploratory

    def train(self, prev_state, action, reward, next_state, done):
        batch_size = self.hyperparameters['batch_size']

        transition = (prev_state, action, reward, next_state, done)
        self.replay_buffer.append(transition)

        if len(self.replay_buffer) < batch_size:
            return
        
        gamma = self.hyperparameters['gamma']
        update_interval = self.hyperparameters['update_interval']

        # Sample a batch of transitions from the replay buffer
        transitions = random.sample(self.replay_buffer, batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

        # Convert to PyTorch tensors
        batch_state = torch.FloatTensor(batch_state)
        batch_next_state = torch.FloatTensor(batch_next_state)
        batch_reward = torch.FloatTensor(batch_reward)
        batch_action = torch.LongTensor(batch_action)
        batch_done = torch.FloatTensor(batch_done)
        
        # Compute current Q values (model predictions)
        current_q_values = self.current_model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        
        # Compute next Q values (target model predictions)
        next_q_values = self.target_model(batch_next_state).detach().max(1)[0]
        target_q_values = batch_reward + gamma * next_q_values * (1 - batch_done)

        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_count % update_interval == 0:
            self.target_model.load_state_dict(self.current_model.state_dict())
        self.steps_count += 1

    def refresh_agent(self):
        self.current_model = DQNNetwork(cart_pole_env.state_size, cart_pole_env.action_size)
        self.target_model = DQNNetwork(cart_pole_env.state_size, cart_pole_env.action_size)
        self.target_model.load_state_dict(self.current_model.state_dict())

    def deserialize_agent(self):
        model_path = self.model_dir / f'{self.hyperparameter_path}.pkl'
        try:
            self.current_model = torch.load(model_path)
            self.target_model = DQNNetwork(cart_pole_env.state_size, cart_pole_env.action_size)
            self.target_model.load_state_dict(self.current_model.state_dict())
        except Exception as e:
            print(f"Failed to load memory from {model_path}: {e}")

    def serialize_agent(self):
        model_path = self.model_dir / f'{self.hyperparameter_path}.pkl'
        torch.save(self.current_model, model_path)