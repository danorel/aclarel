from environments.atari_games.boxing.rl_methods import Agent
from environments.atari_games.boxing.rl_methods.ppo import PPOAgent 
from environments.atari_games.boxing.rl_methods.dqn import DQNAgent

def get_agent(agent_name, curriculum_name, pretrained: bool = False):
    """Factory function to create agent based on the agent_name."""
    agents = {
        'dqn': DQNAgent,
        'ppo': PPOAgent 
    }
    return agents.get(agent_name, DQNAgent)(curriculum_name, pretrained)

def get_curriculum(agent: Agent, min_length=0.25, max_length=0.5):
    """Factory function to fetch the appropriate curriculum."""
    total_evaluations = agent.hyperparameters['total_episodes'] // agent.hyperparameters['evaluation_interval']
    curricula = {
        'baseline': None,
    }
    return curricula.get(agent.metadata['curriculum_name'], None)