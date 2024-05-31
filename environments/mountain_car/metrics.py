import numpy as np

def aar(rewards, lengths, adjustment_factors):
    if not len(rewards) or not len(lengths) or not len(adjustment_factors):
        return 0.0
    adjusted_rewards = [(reward / length) * factor for reward, length, factor in zip(rewards, lengths, adjustment_factors)]
    return np.mean(adjusted_rewards)

def ses(actions_taken, terminal_states):
    safe_actions = sum(1 for (action, is_exploratory), terminal in zip(actions_taken, terminal_states) if not terminal and is_exploratory)
    exploration_actions = sum(1 for action, is_exploratory in actions_taken if is_exploratory)
    return safe_actions / exploration_actions if exploration_actions > 0 else 0

def adjustment_factor(env, episode, total_episodes):
    min_pole_length = 0.25
    max_pole_length = 0.5

    current_pole_length = env.unwrapped.length

    if current_pole_length <= max_pole_length:
        length_adjustment = max_pole_length / current_pole_length
    else:
        length_adjustment = 1

    episode_factor = 1 + (episode / total_episodes) * (min_pole_length / current_pole_length - 1)

    adjustment_factor = length_adjustment * episode_factor
    
    return adjustment_factor