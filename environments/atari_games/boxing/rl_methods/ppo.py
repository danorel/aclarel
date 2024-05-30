import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import pathlib
import random
from torch.cuda.amp import autocast, GradScaler
from collections import deque, namedtuple
import environments.atari_games.boxing.environment as boxing_env
import environments.atari_games.boxing.rl_methods as boxing_rl

agent_name = pathlib.Path(__file__).resolve().stem

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'log_prob'))

class PPONetwork(nn.Module):
    def __init__(self, action_size, hidden_size=64):
        super(PPONetwork, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)

        def conv2d_output_size(input_size, kernel_size, stride, padding=0):
            output_size = (input_size - kernel_size + 2 * padding) // stride + 1
            return output_size

        h, w = 84, 84
        h = conv2d_output_size(h, kernel_size=8, stride=4)
        w = conv2d_output_size(w, kernel_size=8, stride=4)
        h = conv2d_output_size(h, kernel_size=4, stride=2)
        w = conv2d_output_size(w, kernel_size=4, stride=2)
        h = conv2d_output_size(h, kernel_size=3, stride=1)
        w = conv2d_output_size(w, kernel_size=3, stride=1)

        linear_input_size = h * w * 32

        self.actor_fc1 = nn.Linear(linear_input_size, hidden_size)
        self.actor_fc2 = nn.Linear(hidden_size, action_size)
        self.critic_fc1 = nn.Linear(linear_input_size, hidden_size)
        self.critic_fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        actor_x = F.relu(self.actor_fc1(x))
        action_probs = F.softmax(self.actor_fc2(actor_x), dim=-1)

        critic_x = F.relu(self.critic_fc1(x))
        value = self.critic_fc2(critic_x)

        return action_probs, value

class PPOAgent(boxing_rl.Agent):
    def __init__(self, curriculum_name, use_pretrained: bool = False):
        super().__init__(agent_name, curriculum_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        self.autocast = False
        print(f"Autocast gradients: {self.autocast}")
        self.hyperparameters = {
            "total_episodes": 4,
            "alpha": 0.002,
            "gamma": 0.99,
            "clip_epsilon": 0.2,
            "gae_lambda": 0.95,
            "replay_buffer_size": 1000000,
            "batch_size": 2048,
            "print_interval": 1,
            "evaluation_interval": 1,
        }
        self.hyperparameter_path = f"alpha-{self.hyperparameters['alpha']}_gamma-{self.hyperparameters['gamma']}_episodes-{self.hyperparameters['total_episodes']}"
        self.model: PPONetwork = None
        if use_pretrained:
            self.deserialize_agent()
        else:
            self.refresh_agent()

        self.replay_buffer = deque(maxlen=self.hyperparameters['replay_buffer_size'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hyperparameters['alpha'])

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((110, 84)),
            transforms.CenterCrop((84, 84)),
            transforms.ToTensor(),
        ])
        self.steps_count = 0

    def preprocess_single(self, state):
        state = self.transform(state)
        return state.to(self.device)
    
    def preprocess_batch(self, states):
        state_tensors = torch.stack([self.transform(state) for state in states])
        return state_tensors.to(self.device)
    
    def act(self, state, greedily: bool = False):
        state_tensor = self.preprocess_single(state).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.model(state_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), dict(is_exploratory=False, log_prob=log_prob) 

    def train(self, prev_state, action, reward, next_state, done, log_prob):
        self.replay_buffer.append(Transition(prev_state, action, reward, next_state, done, log_prob))
        
        if len(self.replay_buffer) < self.hyperparameters['batch_size']:
            return

        transitions = random.sample(self.replay_buffer, self.hyperparameters['batch_size'])
        states, actions, rewards, next_states, dones, log_probs = zip(*transitions)

        state_batch = self.preprocess_batch(states)
        next_state_batch = torch.zeros_like(state_batch)
        action_batch = torch.tensor(actions, dtype=torch.long, device=self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        non_final_mask = torch.tensor([not done for done in dones], dtype=torch.bool, device=self.device)
        next_state_batch[non_final_mask] = self.preprocess_batch([s for s, done in zip(next_states, dones) if not done])

        action_probs, values = self.model(state_batch)
        probs_distribution = torch.distributions.Categorical(action_probs)
        value_preds = values.squeeze(-1)

        returns = torch.zeros(self.hyperparameters['batch_size'], device=self.device)
        advantages = torch.zeros(self.hyperparameters['batch_size'], device=self.device)
        next_values = torch.zeros(self.hyperparameters['batch_size'], device=self.device)
        if non_final_mask.any():
            _, next_value_preds = self.model(next_state_batch[non_final_mask])
            next_values[non_final_mask] = next_value_preds.detach().squeeze(-1)

        deltas = reward_batch.squeeze(-1) + self.hyperparameters['gamma'] * next_values - value_preds
        last_gae_lam = 0
        for t in reversed(range(self.hyperparameters['batch_size'])):
            last_gae_lam = deltas[t] + self.hyperparameters['gamma'] * self.hyperparameters['gae_lambda'] * last_gae_lam * (1 - int(non_final_mask[t]))
            advantages[t] = last_gae_lam
        returns = advantages + value_preds

        old_log_probs_batch = torch.tensor(log_probs, dtype=torch.float, device=self.device).detach()
        new_log_probs = probs_distribution.log_prob(action_batch).squeeze(-1)
        ratio = torch.exp(new_log_probs - old_log_probs_batch)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.hyperparameters['clip_epsilon'], 1 + self.hyperparameters['clip_epsilon']) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(value_preds, returns)
        total_loss = policy_loss + 0.5 * value_loss

        self.writer.add_scalar('Loss/total_loss', total_loss.item(), self.steps_count)
        self.writer.add_scalar('Loss/policy_loss', policy_loss.item(), self.steps_count)
        self.writer.add_scalar('Loss/value_loss', value_loss.item(), self.steps_count)

        if self.autocast:
            self.optimizer.zero_grad()
            with autocast():
                scaler = GradScaler()
                scaler.scale(total_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
        else:
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
        
        self.writer.add_scalar('Performance/Reward', torch.mean(reward_batch).item(), self.steps_count)
        self.steps_count += 1

    def refresh_agent(self):
        self.model = PPONetwork(boxing_env.env.action_space.n).to(self.device)

    def serialize_agent(self):
        model_path = self.model_dir / f'{self.hyperparameter_path}.pth'
        torch.save(self.model.state_dict(), model_path)

    def deserialize_agent(self):
        model_path = self.model_dir / f'{self.hyperparameter_path}.pth'
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
