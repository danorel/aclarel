import numpy as np

import environments.cart_pole.environment as cart_pole

def logarithmic(base_length=0.5, max_length=1.0):
    def curriculum(env, episode, total_episodes, **metrics):
        scale = np.log(total_episodes)
        new_length = base_length + (max_length - base_length) * (np.log(episode+1) / scale)
        cart_pole.update_env_parameters(env, length=new_length)
    return curriculum