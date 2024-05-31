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
    min_gravity = 0.0025
    max_gravity = 0.025

    current_gravity = env.unwrapped.gravity

    if current_gravity <= max_gravity:
        length_adjustment = max_gravity / current_gravity
    else:
        length_adjustment = 1

    episode_factor = 1 + (episode / total_episodes) * (min_gravity / current_gravity - 1)

    adjustment_factor = length_adjustment * episode_factor
    
    return adjustment_factor