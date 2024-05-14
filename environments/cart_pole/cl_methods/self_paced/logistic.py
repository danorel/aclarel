import numpy as np
import environments.cart_pole.environment as cart_pole

def logistic(base_length=0.5, max_length=0.5, growth_rate=10):
    def curriculum(env, episode, total_episodes):
        episode = max(episode, 1) # avoid division by zero
        x = (episode / total_episodes) * growth_rate
        sigmoid = 1 / (1 + np.exp(-x + growth_rate / 2))
        new_length = base_length + (max_length - base_length) * sigmoid
        cart_pole.update_env_parameters(env, length=new_length)
    return curriculum