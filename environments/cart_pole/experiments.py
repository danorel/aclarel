import environments.cart_pole.cl_methods.pre_defined as pre_defined 
import environments.cart_pole.cl_methods.self_paced as self_paced
import environments.cart_pole.cl_methods.anti_curriculum as anti_curriculum
 
from environments.cart_pole.rl_methods.q_table import QLearningAgent
from environments.cart_pole.rl_methods.dqn import DQNAgent

def get_agent(agent_name, curriculum_name, pretrained: bool = False):
    """Factory function to create agent based on the agent_name."""
    agents = {
        'dqn': DQNAgent,
        'q-learning': QLearningAgent
    }
    return agents.get(agent_name, DQNAgent)(curriculum_name, pretrained)

def get_curriculum(curriculum_name, base_length = 0.25):
    """Factory function to fetch the appropriate curriculum."""
    curricula = {
        'baseline': None,
        'root-p': pre_defined.root_p(base_length),
        'one-pass': pre_defined.one_pass(base_length),
        'hard': self_paced.hard(base_length),
        'linear': self_paced.linear(base_length),
        'logarithmic': self_paced.logarithmic(base_length),
        'logistic': self_paced.logistic(base_length),
        'mixture': self_paced.mixture(base_length),
        'polynomial': self_paced.polynomial(base_length),
        'anti-curriculum': anti_curriculum.anti_curriculum_learning(base_length)
    }
    return curricula.get(curriculum_name, None)