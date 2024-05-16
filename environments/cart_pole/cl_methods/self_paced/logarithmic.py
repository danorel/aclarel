import numpy as np

import environments.cart_pole.environment as cart_pole

def logarithmic(min_length=0.25, max_length=0.5):
    def curriculum(env, evaluation, total_evaluations, **metrics):
        scale = np.log(total_evaluations)
        new_length = min_length + (max_length - min_length) * (np.log(evaluation + 1) / scale)
        cart_pole.update_env_parameters(env, length=new_length)
    return curriculum