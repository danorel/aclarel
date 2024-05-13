import numpy as np

import environments.cart_pole.environment as cart_pole

def root_p(base_length=0.5, max_length=0.5, exponent=2):
    def curriculum(env, episode, total_episodes):
        # Linearly increase the pole length over episodes
        if episode == 0:
            new_length = base_length
        else:
            new_length = base_length + (max_length - base_length) * np.sqrt(episode) / (exponent * np.sqrt(episode))
            cart_pole.update_env_parameters(env, length=new_length)
    return curriculum