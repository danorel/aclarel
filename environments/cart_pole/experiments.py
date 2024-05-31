import environments.cart_pole.cl_methods.pre_defined as pre_defined 
import environments.cart_pole.cl_methods.self_paced as self_paced
import environments.cart_pole.cl_methods.anti_curriculum as anti_curriculum
import environments.cart_pole.cl_methods.teacher_learning as teacher_learning
import environments.cart_pole.cl_methods.transfer_learning as transfer_learning
 
from environments.cart_pole.rl_methods import Agent
from environments.cart_pole.rl_methods.q_learning import QLearningAgent
from environments.cart_pole.rl_methods.dqn import DQNAgent

def get_agent(agent_name, curriculum_name, pretrained: bool = False):
    """Factory function to create agent based on the agent_name."""
    agents = {
        'dqn': DQNAgent,
        'q-learning': QLearningAgent
    }
    return agents.get(agent_name, DQNAgent)(curriculum_name, pretrained)

def get_curriculum(agent: Agent, min_length=0.25, max_length=0.5):
    """Factory function to fetch the appropriate curriculum."""
    total_evaluations = agent.hyperparameters['total_episodes'] // agent.hyperparameters['evaluation_interval']
    curricula = {
        'baseline': None,
        'teacher-learning': teacher_learning.teacher_student_curriculum(min_length, max_length, min_reward=0.0, max_reward=500.0),
        'transfer-learning': transfer_learning.transfer_learning_curriculum(min_length, max_length, source_evaluations=total_evaluations // 2, target_evaluations=total_evaluations),
        'root-p': pre_defined.root_p(min_length, max_length, exponent=2),
        'one-pass': pre_defined.one_pass(min_length, max_length, total_evaluations),
        'hard': self_paced.hard(min_length, max_length, milestones=[0.25, 0.5, 0.75]),
        'linear': self_paced.linear(min_length, max_length),
        'logarithmic': self_paced.logarithmic(min_length, max_length),
        'logistic': self_paced.logistic(min_length, max_length),
        'mixture': self_paced.mixture(min_length, max_length, switch=0.5),
        'polynomial': self_paced.polynomial(min_length, max_length),
        'anti-curriculum': anti_curriculum.anti_curriculum_learning(min_length, max_length, total_evaluations)
    }
    return curricula.get(agent.metadata['curriculum_name'], None)