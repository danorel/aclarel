import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import pathlib
import random
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from collections import deque, namedtuple
import environments.atari_games.boxing.environment as boxing_env
import environments.atari_games.boxing.rl_methods as boxing_rl

agent_name = pathlib.Path(__file__).resolve().stem

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class DQNNetwork(nn.Module):
    def __init__(self, action_size, hidden_size = 64):
        super(DQNNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=8, stride=5)
        self.conv2 = nn.Conv2d(24, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)

        def conv_output_size(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1

        convw = conv_output_size(conv_output_size(conv_output_size(boxing_env.env.observation_space.shape[0], 8, 5), 4, 2), 3, 1)
        convh = conv_output_size(conv_output_size(conv_output_size(boxing_env.env.observation_space.shape[1], 8, 5), 4, 2), 3, 1)
        
        linear_input_size = convw * convh * 32

        self.fc1 = nn.Linear(linear_input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
    
    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DQNAgent(boxing_rl.Agent):
    def __init__(self, curriculum_name, use_pretrained: bool = False):
        super().__init__(agent_name, curriculum_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        self.hyperparameters = {
            "total_episodes": 100,
            "alpha": 0.00025,
            "gamma": 0.99,
            "replay_buffer_size": 1000000,
            "batch_size": 256,
            "initial_epsilon": 1.0,
            "minimum_epsilon": 0.01,
            "epsilon_decay": 0.995,
            "print_interval": 50,
            "evaluation_interval": 10,
            "update_interval": 10000
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
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        batch_size = self.hyperparameters['batch_size']
        state_shape = (1, boxing_env.env.observation_space.shape[0], boxing_env.env.observation_space.shape[1])
        action_shape = (batch_size, 1)
        reward_shape = (batch_size, 1)
        self.state_batch = torch.zeros((batch_size,) + state_shape, dtype=torch.float32, device=self.device)
        self.action_batch = torch.zeros(action_shape, dtype=torch.long, device=self.device)
        self.reward_batch = torch.zeros(reward_shape, dtype=torch.float32, device=self.device)
        self.next_state_batch = torch.zeros((batch_size,) + state_shape, dtype=torch.float32, device=self.device)
        self.non_final_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

    def preprocess_single(self, state):
        return self.transform(state).to(self.device)
    
    def preprocess_batch(self, states):
        return torch.stack([self.transform(state) for state in states]).to(self.device)

    def act(self, state, greedily: bool = False):
        is_exploratory = False

        if greedily:
            with torch.no_grad():
                state_tensor = self.preprocess_single(state).unsqueeze(0)
                action_values = self.current_model(state_tensor)
                action = torch.argmax(action_values).item()
        else:
            epsilon_decay = self.hyperparameters['epsilon_decay']
            minimum_epsilon = self.hyperparameters['minimum_epsilon']

            if np.random.random() < self.epsilon:
                is_exploratory = True
                action = boxing_env.env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = self.preprocess_single(state).unsqueeze(0)
                    action_values = self.current_model(state_tensor)
                    action = torch.argmax(action_values).item()

            self.epsilon *= epsilon_decay
            self.epsilon = max(minimum_epsilon, self.epsilon)
        
        return action, dict(is_exploratory=is_exploratory, log_prob=None)

    def train(self, prev_state, action, reward, next_state, done, log_prob):
        self.replay_buffer.append(Transition(prev_state, action, reward, next_state, done))
        if len(self.replay_buffer) < self.hyperparameters['batch_size']:
            return

        transitions = random.sample(self.replay_buffer, self.hyperparameters['batch_size'])
        states, actions, rewards, next_states, dones = zip(*transitions)

        self.state_batch = self.preprocess_batch(states)
        self.action_batch = torch.tensor(actions, dtype=torch.long, device=self.device)
        self.reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        self.non_final_mask = torch.tensor([not done for done in dones], dtype=torch.bool, device=self.device)
        non_final_next_states = [s for s, done in zip(next_states, dones) if not done]
        if non_final_next_states:
            non_final_next_states = self.preprocess_batch(non_final_next_states)

        state_action_values = self.current_model(self.state_batch).gather(1, self.action_batch.unsqueeze(1)).squeeze(-1)

        next_state_values = torch.zeros(self.hyperparameters['batch_size'], device=self.device)
        next_state_values[self.non_final_mask] = self.target_model(self.next_state_batch).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.hyperparameters['gamma']) + self.reward_batch 

        if self.device == 'cuda':
            scaler = GradScaler()
            with autocast():
                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
        else:
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.current_model.parameters(), 1)
            self.optimizer.step()

        if self.steps_count % self.hyperparameters['update_interval'] == 0:
            self.target_model.load_state_dict(self.current_model.state_dict())
        self.steps_count += 1

    def refresh_agent(self):
        self.current_model = DQNNetwork(boxing_env.action_size).to(self.device)
        self.target_model = DQNNetwork(boxing_env.action_size).to(self.device)
        self.target_model.load_state_dict(self.current_model.state_dict())

    def serialize_agent(self):
        model_path = self.model_dir / f'{self.hyperparameter_path}.pth'
        torch.save(self.current_model.state_dict(), model_path)

    def deserialize_agent(self):
        model_path = self.model_dir / f'{self.hyperparameter_path}.pth'
        self.current_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.target_model.load_state_dict(self.current_model.state_dict())