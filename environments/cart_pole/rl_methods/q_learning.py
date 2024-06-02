import numpy as np
import torch
import pathlib
import environments.cart_pole.environment as cart_pole_env
import environments.cart_pole.rl_methods as cart_pole_rl

agent_name = pathlib.Path(__file__).resolve().stem

amount_of_bins = 20
state_bins = [
    np.linspace(-4.8, 4.8, amount_of_bins),
    np.linspace(-4, 4, amount_of_bins),
    np.linspace(-.418, .418, amount_of_bins),
    np.linspace(-4, 4, amount_of_bins)
]

def get_discrete_state(state):
    state_index = [
        np.digitize(state[i], state_bins[i]) - 1
        for i in range(len(cart_pole_env.env.observation_space.high))
    ]
    return tuple(state_index)

class QLearningAgent(cart_pole_rl.Agent):
    def __init__(self, curriculum_name, use_pretrained: bool = False):
        super().__init__(agent_name, curriculum_name)
        self.hyperparameters = {
            "total_episodes": 50000,
            "alpha": 0.1,
            "gamma": 0.95,
            "initial_epsilon": 1.0,
            "minimum_epsilon": 0.005,
            "epsilon_decay": 0.99995,
            "print_interval": 250,
            "evaluation_interval": 10,
            "train_interval": 10,
            'log_interval': 1,
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
                action = cart_pole_env.env.action_space.sample()
            else:
                action = np.argmax(self.q_table[get_discrete_state(state)])
        return action, is_exploratory
    
    def train(self, prev_state, action, reward, next_state, done):
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

    def refresh_agent(self):
        states = tuple(len(bins) + 1 for bins in state_bins)
        actions = (cart_pole_env.env.action_space.n,)
        self.q_table = np.random.uniform(low=-2, high=0, size=(states + actions))

    def deserialize_agent(self):
        model_path = self.model_dir / f'{self.hyperparameter_path}.pt'
        self.q_table = torch.load(model_path, map_location=self.device)

    def serialize_agent(self):
        model_path = self.model_dir / f'{self.hyperparameter_path}.pt'
        torch.save(self.q_table, model_path)