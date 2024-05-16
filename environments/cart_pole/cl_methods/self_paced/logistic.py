import numpy as np
import environments.cart_pole.environment as cart_pole

def logistic(min_length=0.5, max_length=0.5, growth_rate=10):
    def curriculum(env, evaluation, total_evaluations, **metrics):
        evaluation = max(evaluation, 1) # avoid division by zero
        progress = (evaluation / total_evaluations) * growth_rate
        sigmoid = 1 / (1 + np.exp(-progress + growth_rate / 2))
        new_length = min_length + (max_length - min_length) * sigmoid
        cart_pole.update_env_parameters(env, length=new_length)
    return curriculum