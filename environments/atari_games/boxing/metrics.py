import numpy as np

def aar(rewards, lengths, adjustment_factors):
    if not len(rewards) or not len(lengths) or not len(adjustment_factors):
        return 0.0
    # Adjust rewards by episode length and difficulty adjustment factor
    adjusted_rewards = [(reward / length) * factor for reward, length, factor in zip(rewards, lengths, adjustment_factors)]
    # Return the average of these adjusted rewards
    return np.mean(adjusted_rewards)

def ses(actions_taken, terminal_states):
    safe_actions = sum(1 for (action, is_exploratory), terminal in zip(actions_taken, terminal_states) if not terminal and is_exploratory)
    exploration_actions = sum(1 for action, is_exploratory in actions_taken if is_exploratory)
    return safe_actions / exploration_actions if exploration_actions > 0 else 0

def adjustment_factor(env, episode, total_episodes):
    # Constants for adjustment calculations
    min_difficulty = 1  # Minimum difficulty level
    max_difficulty = 10  # Maximum difficulty level

    # Gradually increase the difficulty of the opponent
    episode_factor = min_difficulty + (max_difficulty - min_difficulty) * (episode / total_episodes)

    current_difficulty = 1

    # Calculate a baseline adjustment based on pole length
    if current_difficulty <= max_difficulty:
        length_adjustment = max_difficulty / current_difficulty
    else:
        length_adjustment = 1

    # Combine the two adjustments
    adjustment_factor = length_adjustment * episode_factor
    
    return adjustment_factor