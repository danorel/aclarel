import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity
import pathlib
import environments.cart_pole.environment as cart_pole_env
import environments.cart_pole.rl_methods as cart_pole_rl

agent_name = pathlib.Path(__file__).resolve().stem

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

profiler_settings = {
    "schedule": torch.profiler.schedule(wait=1, warmup=1, active=3),
    "on_trace_ready": torch.profiler.tensorboard_trace_handler('./runs'),
    "record_shapes": True,
    "profile_memory": True,
    "with_stack": True,
    "activities": [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == 'cuda' else []),
}

amount_of_bins = 20
state_bins = [
    np.linspace(-4.8, 4.8, amount_of_bins),     # Position
    np.linspace(-4, 4, amount_of_bins),         # Velocity
    np.linspace(-0.418, 0.418, amount_of_bins), # Angle
    np.linspace(-4, 4, amount_of_bins)          # Angular velocity
]

def get_discrete_state(state):
    index = [np.digitize(val, bins) - 1 for val, bins in zip(state, state_bins)]
    return tuple(index)

class QLearningAgent(cart_pole_rl.Agent):
    def __init__(self, curriculum_name, use_pretrained: bool = False):
        super().__init__(agent_name, curriculum_name)
        self.device = device
        print(f"Device: {self.device}")
        self.hyperparameters = {
            "total_episodes": 50000,
            "alpha": 0.1,
            "gamma": 0.99,
            "initial_epsilon": 0.1,
            "minimum_epsilon": 0.005,
            "epsilon_decay": 0.9999,
            "print_interval": 250,
            "evaluation_interval": 10,
        }
        self.hyperparameter_path = f"alpha-{self.hyperparameters['alpha']}_gamma-{self.hyperparameters['gamma']}_episodes-{self.hyperparameters['total_episodes']}"
        if use_pretrained:
            self.deserialize_agent()
        else:
            self.refresh_agent()
        self.epsilon = self.hyperparameters['initial_epsilon']

    def act(self, state, greedily: bool = False):
        is_exploratory = False
        if greedily:
            action = torch.argmax(self.q_table[get_discrete_state(state)]).item()
        else:
            if np.random.random() < self.epsilon:
                is_exploratory = True
                action = cart_pole_env.env.action_space.sample()
            else:
                action = torch.argmax(self.q_table[get_discrete_state(state)]).item()
            self.epsilon *= self.hyperparameters['epsilon_decay']
            self.epsilon = max(self.hyperparameters['minimum_epsilon'], self.epsilon)
        return action, is_exploratory
    
    def train(self, prev_state, action, reward, next_state, done):
        with profile(**profiler_settings) as prof:
            self.steps_count += 1

            prev_state = torch.tensor([get_discrete_state(prev_state)], device=self.device, dtype=torch.long)
            next_state = torch.tensor([get_discrete_state(next_state)], device=self.device, dtype=torch.long)
            action = torch.tensor([action], device=self.device, dtype=torch.long)

            current_q_value = self.q_table[prev_state, action].squeeze()
            max_next_q_value = torch.max(self.q_table[next_state]).detach()

            new_q_value = current_q_value + self.hyperparameters['alpha'] * (reward + self.hyperparameters['gamma'] * max_next_q_value - current_q_value)
            
            td_error = new_q_value - current_q_value
            self.q_table[prev_state, action] = new_q_value

            td_error_mean = td_error.mean().item()
            self.writer.add_scalar('Loss/TD_Error_Mean', td_error_mean, self.steps_count)

            td_error_sum = td_error.sum().item()
            self.writer.add_scalar('Loss/TD_Error_Sum', td_error_sum, self.steps_count)

            prof.step()

    def refresh_agent(self):
        states = tuple(len(bins) + 1 for bins in state_bins)
        actions = (cart_pole_env.env.action_space.n,)
        self.q_table = torch.zeros(states + actions, dtype=torch.float32, device=self.device)

    def deserialize_agent(self):
        model_path = self.model_dir / f'{self.hyperparameter_path}.pt'
        self.q_table = torch.load(model_path, map_location=self.device)

    def serialize_agent(self):
        model_path = self.model_dir / f'{self.hyperparameter_path}.pt'
        torch.save(self.q_table, model_path)