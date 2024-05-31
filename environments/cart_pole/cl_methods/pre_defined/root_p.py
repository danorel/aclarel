import numpy as np

import environments.cart_pole.environment as cart_pole

def root_p(min_length=0.25, max_length=0.5, exponent=2):
    def curriculum(env, evaluation, total_evaluations, **metrics):
        evaluation = max(evaluation, 1)
        sqrt_evaluation = np.sqrt(evaluation)
        new_length = min_length + (max_length - min_length) * sqrt_evaluation / (exponent * sqrt_evaluation)
        cart_pole.update_env_parameters(env, length=new_length)
    return curriculum