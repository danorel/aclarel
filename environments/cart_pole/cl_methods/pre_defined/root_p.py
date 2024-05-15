import numpy as np

import environments.cart_pole.environment as cart_pole

def root_p(base_length=0.5, max_length=0.5, exponent=2):
    def curriculum(env, episode, total_episodes, **metrics):
        episode = max(episode, 1) # avoid division by zero
        sqrt_episode = np.sqrt(episode)
        new_length = base_length + (max_length - base_length) * sqrt_episode / (exponent * sqrt_episode)
        cart_pole.update_env_parameters(env, length=new_length)
    return curriculum