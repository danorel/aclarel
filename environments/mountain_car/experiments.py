import environments.mountain_car.cl_methods.pre_defined as pre_defined 
import environments.mountain_car.cl_methods.self_paced as self_paced
import environments.mountain_car.cl_methods.anti_curriculum as anti_curriculum
import environments.mountain_car.cl_methods.teacher_learning as teacher_learning
import environments.mountain_car.cl_methods.transfer_learning as transfer_learning
 
from environments.mountain_car.rl_methods import Agent
from environments.mountain_car.rl_methods.q_learning import QLearningAgent
from environments.mountain_car.rl_methods.dqn import DQNAgent

def get_agent(agent_name, curriculum_name, pretrained: bool = False):
    """Factory function to create agent based on the agent_name."""
    agents = {
        'dqn': DQNAgent,
        'q-learning': QLearningAgent
    }
    return agents.get(agent_name, DQNAgent)(curriculum_name, pretrained)

def get_curriculum(agent: Agent, min_gravity=0.00025, max_gravity=0.0025):
    """Factory function to fetch the appropriate curriculum."""
    total_evaluations = agent.hyperparameters['total_episodes'] // agent.hyperparameters['evaluation_interval']
    curricula = {
        'baseline': None,
        'teacher-learning': teacher_learning.teacher_student_curriculum(min_gravity, max_gravity, min_reward=-200, max_reward=-110),
        'transfer-learning': transfer_learning.transfer_learning_curriculum(min_gravity, max_gravity, source_evaluations=total_evaluations // 2, target_evaluations=total_evaluations),
        'root-p': pre_defined.root_p(min_gravity, max_gravity, exponent=2),
        'one-pass': pre_defined.one_pass(min_gravity, max_gravity, total_evaluations),
        'hard': self_paced.hard(min_gravity, max_gravity, milestones=[0.25, 0.5, 0.75]),
        'linear': self_paced.linear(min_gravity, max_gravity),
        'logarithmic': self_paced.logarithmic(min_gravity, max_gravity),
        'logistic': self_paced.logistic(min_gravity, max_gravity),
        'mixture': self_paced.mixture(min_gravity, max_gravity, switch=0.5),
        'polynomial': self_paced.polynomial(min_gravity, max_gravity),
        'anti-curriculum': anti_curriculum.anti_curriculum_learning(min_gravity, max_gravity, total_evaluations)
    }
    return curricula.get(agent.metadata['curriculum_name'], None)