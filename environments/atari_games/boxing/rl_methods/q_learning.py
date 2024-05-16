import numpy as np
import pathlib
import environments.atari_games.boxing.environment as boxing_env
import environments.atari_games.boxing.rl_methods as boxing_rl

agent_name = pathlib.Path(__file__).resolve().stem

class QLearningAgent(boxing_rl.Agent):
    def __init__(self, curriculum_name, use_pretrained: bool = False):
        super().__init__(agent_name, curriculum_name)
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
        # Define Q-Learning architecture
        if use_pretrained:
            self.deserialize_agent()
        else:
            self.refresh_agent()
        # Parameters
        self.epsilon = self.hyperparameters['initial_epsilon']

    def act(self, state, greedily: bool = False):
        is_exploratory = False

        if greedily:
            action = np.argmax(self.q_table[state])
        else:
            epsilon_decay = self.hyperparameters['epsilon_decay']
            minimum_epsilon = self.hyperparameters['minimum_epsilon']

            if np.random.random() < self.epsilon:
                is_exploratory = True
                action = boxing_env.env.action_space.sample()
            else:
                is_exploratory = False
                action = np.argmax(self.q_table[state])

            self.epsilon *= epsilon_decay
            self.epsilon = max(minimum_epsilon, self.epsilon)

        return action, is_exploratory
    
    def train(self, prev_state, action, reward, next_state, done):
        alpha = self.hyperparameters['alpha']
        gamma = self.hyperparameters['gamma']
        self.q_table[prev_state + (action,)] = self.q_table[prev_state + (action,)] + alpha * (reward + gamma * np.max(self.q_table[next_state]) - self.q_table[prev_state + (action,)])
    
    def refresh_agent(self):
        self.q_table = np.zeros(
            (boxing_env.action_size, boxing_env.state_size), 
            dtype=np.float32
        )

    def deserialize_agent(self):
        model_path = self.model_dir / f'{self.hyperparameter_path}.npy' 
        self.q_table = np.load(model_path, allow_pickle=False)

    def serialize_agent(self):
        model_path = self.model_dir / f'{self.hyperparameter_path}.npy'
        np.save(model_path, self.q_table)