import numpy as np

import environments.cart_pole.environment as cart_pole

def transfer_learning_curriculum(source_length=0.5, target_length=1.0, source_episodes=50, total_episodes=100):
    source_lengths = np.linspace(source_length, target_length, source_episodes)
    target_lengths = [target_length] * (total_episodes - source_episodes)

    def curriculum(env, episode, total_episodes, **metrics):
        if episode < source_episodes:
            index = min(episode, source_episodes - 1)
            new_length = source_lengths[index]
        else:
            index = min(episode - source_episodes, total_episodes - source_episodes - 1)
            new_length = target_lengths[index]
        cart_pole.update_env_parameters(env, length=new_length)
    
    return curriculum