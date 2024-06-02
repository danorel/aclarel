import environments.atari_games.boxing.cl_methods.pre_defined as pre_defined 
import environments.atari_games.boxing.cl_methods.self_paced as self_paced
import environments.atari_games.boxing.cl_methods.anti_curriculum as anti_curriculum
import environments.atari_games.boxing.cl_methods.teacher_learning as teacher_learning
import environments.atari_games.boxing.cl_methods.transfer_learning as transfer_learning
 
from environments.atari_games.boxing.rl_methods import Agent
from environments.atari_games.boxing.rl_methods.dqn import DQNAgent
from environments.atari_games.boxing.rl_methods.ppo import PPOAgent

def get_agent(agent_name, curriculum_name, pretrained: bool = False):
    """Factory function to create agent based on the agent_name."""
    agents = {
        'dqn': DQNAgent,
        'ppo': PPOAgent,
    }
    return agents.get(agent_name, DQNAgent)(curriculum_name, pretrained)

def get_curriculum(agent: Agent, min_frame_skip=1, max_frame_skip=10):
    """Factory function to fetch the appropriate curriculum."""
    total_evaluations = agent.hyperparameters['total_episodes'] // agent.hyperparameters['evaluation_interval']
    curricula = {
        'baseline': None,
        'teacher-learning': teacher_learning.teacher_student_curriculum(min_frame_skip, max_frame_skip, min_reward=-100, max_reward=100),
        'transfer-learning': transfer_learning.transfer_learning_curriculum(min_frame_skip, max_frame_skip, source_frame_skip=1, target_frame_skip=4, source_evaluations=total_evaluations // 2, target_evaluations=total_evaluations),
        'root-p': pre_defined.root_p(min_frame_skip, max_frame_skip, exponent=2),
        'one-pass': pre_defined.one_pass(min_frame_skip, max_frame_skip, total_evaluations),
        'hard': self_paced.hard(min_frame_skip, max_frame_skip, milestones=[0.25, 0.5, 0.75]),
        'linear': self_paced.linear(min_frame_skip, max_frame_skip),
        'logarithmic': self_paced.logarithmic(min_frame_skip, max_frame_skip),
        'logistic': self_paced.logistic(min_frame_skip, max_frame_skip),
        'mixture': self_paced.mixture(min_frame_skip, max_frame_skip, switch=0.5),
        'polynomial': self_paced.polynomial(min_frame_skip, max_frame_skip),
        'anti-curriculum': anti_curriculum.anti_curriculum_learning(min_frame_skip, max_frame_skip, total_evaluations)
    }
    return curricula.get(agent.metadata['curriculum_name'], None)