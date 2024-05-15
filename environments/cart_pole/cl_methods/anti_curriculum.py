import numpy as np

import environments.cart_pole.environment as cart_pole

def anti_curriculum_learning(base_length=0.5, max_length=1.0, total_episodes=100):
    lengths = np.linspace(max_length, base_length, total_episodes)

    def curriculum(env, episode, total_episodes, **metrics):
        index = min(episode, total_episodes - 1)
        new_length = lengths[index]
        cart_pole.update_env_parameters(env, length=new_length)
    
    return curriculum