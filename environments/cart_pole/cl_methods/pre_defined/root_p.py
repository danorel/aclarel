import numpy as np
import pathlib
import environments.cart_pole.environment as cart_pole
from environments.cart_pole.rl_methods.q_table import QLearningAgent
from environments.cart_pole.rl_methods.dqn import DQNAgent

def root_p(base_length=0.5, max_length=0.5, exponent=2):
    def curriculum(env, episode, total_episodes):
        # Linearly increase the pole length over episodes
        if episode >= 0 and episode <= total_episodes:
            new_length = base_length + (max_length - base_length) * np.sqrt(episode) / (exponent * np.sqrt(episode))
            cart_pole.update_env_parameters(env, length=new_length)
    return curriculum

if __name__ == "__main__":
    cart_pole.train_evaluate(
        agent=DQNAgent(curriculum_name=pathlib.Path(__file__).resolve().stem), 
        curriculum=root_p(base_length=0.25)
    )