import torch
import torch.nn as nn
import torch.optim as optim
import pathlib
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
    def __init__(self, curriculum_name):
        super().__init__(agent_name, curriculum_name)
        if not cart_pole_env.USE_PRETRAINED_MODEL:
            self.refresh_agent()
        else:
            self.deserialize_agent()
        self.memory = deque(maxlen=10000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cart_pole_env.alpha)
    
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_values = self.model(state)
        return torch.argmax(action_values).item()

    def train(self, prev_state, action, reward, next_state):
        prev_state = torch.FloatTensor(prev_state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        reward = torch.FloatTensor([reward])
        action = torch.LongTensor([action])
        
        # Get predicted Q-values for current state
        pred_q = self.model(prev_state).gather(1, action.unsqueeze(1))
        
        # Get maximum Q-value for next state from target model
        next_max_q = self.model(next_state).detach().max(1)[0]
        target_q = reward + (cart_pole_env.gamma * next_max_q)
        
        # Compute loss
        loss = nn.MSELoss()(pred_q.squeeze(), target_q)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def refresh_agent(self):
        self.model = DQNNetwork(cart_pole_env.state_size, cart_pole_env.action_size)

    def deserialize_agent(self):
        model_path = self.model_dir / f'{self.parameters}.pkl'
        try:
            self.model = torch.load(model_path)
        except Exception as e:
            print(f"Failed to load memory from {self.path}: {e}")

    def serialize_agent(self):
        model_path = self.model_dir / f'{self.parameters}.pkl'
        torch.save(self.model, model_path)