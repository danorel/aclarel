import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity
import pathlib
import environments.mountain_car.environment as mountain_car_env
import environments.mountain_car.rl_methods as mountain_car_rl

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

amount_of_bins = 30
state_bins = [
    np.linspace(-1.2, 0.6, amount_of_bins),  # Position from -1.2 to 0.6
    np.linspace(-0.07, 0.07, amount_of_bins) # Velocity from -0.07 to 0.07
]

def get_discrete_state(state):
    index = [np.digitize(val, bins) - 1 for val, bins in zip(state, state_bins)]
    return tuple(index)

class QLearningAgent(mountain_car_rl.Agent):
    def __init__(self, curriculum_name, use_pretrained: bool = False):
        super().__init__(agent_name, curriculum_name)
        self.device = device
        print(f"Device: {self.device}")
        self.hyperparameters = {
            "total_episodes": 3000,
            "alpha": 0.1,
            "gamma": 0.99,
            "initial_epsilon": 1.0,
            "minimum_epsilon": 0.01,
            "epsilon_decay": 0.99999,
            "print_interval": 10,
            "evaluation_interval": 30,
            "train_interval": 10,
            "log_interval": 500,
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
            action = np.argmax(self.q_table[get_discrete_state(state)])
        else:
            if np.random.random() < self.epsilon:
                is_exploratory = True
                action = mountain_car_env.env.action_space.sample()
            else:
                action = np.argmax(self.q_table[get_discrete_state(state)])
        return action, dict(log_prob=None, is_exploratory=is_exploratory)
    
    def train(self, prev_state, action, reward, next_state, done, log_prob):
        with profile(**profiler_settings) as prof:
            self.steps_count += 1

            prev_state_idx = get_discrete_state(prev_state)
            next_state_idx = get_discrete_state(next_state)

            current_q_value = self.q_table[prev_state_idx, action]
            max_next_q_value = np.max(self.q_table[next_state_idx])

            new_q_value = (1 - self.hyperparameters['alpha']) * current_q_value + self.hyperparameters['alpha'] * (
                reward + self.hyperparameters['gamma'] * max_next_q_value)

            self.q_table[prev_state_idx, action] = new_q_value

            if self.steps_count % self.hyperparameters['train_interval'] == 0:
                self.epsilon *= self.hyperparameters['epsilon_decay']
                self.epsilon = max(self.hyperparameters['minimum_epsilon'], self.epsilon)

            if self.steps_count % self.hyperparameters['log_interval'] == 0:
                td_error = new_q_value - current_q_value
                td_error_mean = td_error.mean().item()
                self.writer.add_scalar('Loss/TD_Error_Mean', td_error_mean, self.steps_count)
                td_error_sum = td_error.sum().item()
                self.writer.add_scalar('Loss/TD_Error_Sum', td_error_sum, self.steps_count)
                prof.step()

    def refresh_agent(self):
        states = tuple(len(bins) + 1 for bins in state_bins)
        actions = (mountain_car_env.env.action_space.n,)
        self.q_table = np.zeros(states + actions, dtype=np.float32)

    def deserialize_agent(self):
        model_path = self.model_dir / f'{self.hyperparameter_path}.pt'
        self.q_table = torch.load(model_path, map_location=self.device)

    def serialize_agent(self):
        model_path = self.model_dir / f'{self.hyperparameter_path}.pt'
        torch.save(self.q_table, model_path)