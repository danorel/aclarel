import numpy as np
import pathlib
import environments.cart_pole.environment as cart_pole_env
import environments.cart_pole.rl_methods as cart_pole_rl

agent_name = pathlib.Path(__file__).resolve().stem

class QLearningAgent(cart_pole_rl.Agent):
    def __init__(self, curriculum_name):
        super().__init__(agent_name, curriculum_name)
        if not cart_pole_env.USE_PRETRAINED_MODEL:
            self.refresh_agent()
        else:
            self.deserialize_agent()

    def act(self, state):
        return np.argmax(self.q_table[state])
    
    def train(self, prev_state, action, reward, next_state):
        self.q_table[prev_state + (action,)] = self.q_table[prev_state + (action,)] + cart_pole_env.alpha * (reward + cart_pole_env.gamma * np.max(self.q_table[next_state]) - self.q_table[prev_state + (action,)])
    
    def refresh_agent(self):
        states = tuple(len(bins) + 1 for bins in cart_pole_env.state_bins)
        actions = (cart_pole_env.action_size,)
        self.q_table = np.zeros(states + actions, dtype=np.float32)

    def deserialize_agent(self):
        model_path = self.model_dir / f'{self.parameters}.npy'
        self.q_table = np.load(model_path, allow_pickle=False)

    def serialize_agent(self):
        model_path = self.model_dir / f'{self.parameters}.npy'
        np.save(model_path, self.q_table)