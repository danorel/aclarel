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

amount_of_bins = 20
state_bins = [
    np.linspace(-1.2, 0.6, amount_of_bins),  # Position from -1.2 to 0.6
    np.linspace(-0.07, 0.07, amount_of_bins) # Velocity from -0.07 to 0.07
]

def get_discrete_state(state):
    index = []
    for i, val in enumerate(state):
        index.append(np.digitize(val, state_bins[i]) - 1)
    return tuple(index)

class QLearningAgent(mountain_car_rl.Agent):
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
        self.total_steps = 0

    def act(self, state, greedily: bool = False):
        state = get_discrete_state(state)
        is_exploratory = False
        state_tensor = torch.tensor(state, device=self.device)
        if greedily:
            action = torch.argmax(self.q_table[state_tensor]).item()
        else:
            if np.random.random() < self.epsilon:
                is_exploratory = True
                action = mountain_car_env.env.action_space.sample()
            else:
                action = torch.argmax(self.q_table[state_tensor]).item()
            self.epsilon *= self.hyperparameters['epsilon_decay']
            self.epsilon = max(self.hyperparameters['minimum_epsilon'], self.epsilon)
        return action, is_exploratory
    
    def train(self, prev_state, action, reward, next_state, done):
        with profile(**profiler_settings) as prof:
            prev_state = torch.tensor([get_discrete_state(prev_state)], device=self.device, dtype=torch.long)
            next_state = torch.tensor([get_discrete_state(next_state)], device=self.device, dtype=torch.long)
            action = torch.tensor([action], device=self.device, dtype=torch.long)

            current_q_value = self.q_table[prev_state, action].squeeze()
            max_next_q_value = torch.max(self.q_table[next_state]).detach()

            new_q_value = current_q_value + self.hyperparameters['alpha'] * (reward + self.hyperparameters['gamma'] * max_next_q_value - current_q_value)
            
            td_error = new_q_value - current_q_value
            self.q_table[prev_state, action] = new_q_value

            td_error_mean = td_error.mean().item()
            self.writer.add_scalar('Loss/TD_Error_Mean', td_error_mean, self.total_steps)

            td_error_sum = td_error.sum().item()
            self.writer.add_scalar('Loss/TD_Error_Sum', td_error_sum, self.total_steps)

            prof.step()

            self.total_steps += 1

    def refresh_agent(self):
        states = tuple(len(bins) + 1 for bins in state_bins)
        actions = (mountain_car_env.env.action_space.n,)
        self.q_table = torch.zeros(states + actions, dtype=torch.float32, device=self.device)

    def deserialize_agent(self):
        model_path = self.model_dir / f'{self.hyperparameter_path}.pt'
        self.q_table = torch.load(model_path, map_location=self.device)

    def serialize_agent(self):
        model_path = self.model_dir / f'{self.hyperparameter_path}.pt'
        torch.save(self.q_table, model_path)