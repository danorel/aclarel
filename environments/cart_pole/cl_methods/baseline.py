import pathlib
import environments.cart_pole.environment as cart_pole
from environments.cart_pole.rl_methods.q_table import QLearningAgent
from environments.cart_pole.rl_methods.dqn import DQNAgent

if __name__ == "__main__":
    cart_pole.train_evaluate(
        agent=DQNAgent(curriculum_name=pathlib.Path(__file__).resolve().stem), 
        curriculum=None
    )