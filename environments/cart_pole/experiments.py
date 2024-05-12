import environments.cart_pole.cl_methods.pre_defined as pre_defined 
import environments.cart_pole.cl_methods.self_paced as self_paced
 
from environments.cart_pole.rl_methods.q_table import QLearningAgent
from environments.cart_pole.rl_methods.dqn import DQNAgent

def get_agent(agent_name, curriculum_name, pretrained: bool = False):
    """Factory function to create agent based on the agent_name."""
    agents = {
        'dqn': DQNAgent,
        'q-learning': QLearningAgent
    }
    return agents.get(agent_name, DQNAgent)(curriculum_name, pretrained)

def get_curriculum(curriculum_name):
    """Factory function to fetch the appropriate curriculum."""
    curricula = {
        'baseline': None,
        'root_p': pre_defined.root_p(base_length=0.25),
        'linear': self_paced.linear(base_length=0.25),
        'logarithmic': self_paced.logarithmic(base_length=0.25)
    }
    return curricula.get(curriculum_name, None)