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
    min_frame_skip = 1
    max_frame_skip = 10

    episode_factor = min_frame_skip + (max_frame_skip - min_frame_skip) * (episode / total_episodes)

    current_frame_skip = env.unwrapped._frameskip

    if current_frame_skip <= max_frame_skip:
        frame_skip_adjustment = min_frame_skip / current_frame_skip
    else:
        frame_skip_adjustment = 1

    adjustment_factor = frame_skip_adjustment * episode_factor
    
    return adjustment_factor