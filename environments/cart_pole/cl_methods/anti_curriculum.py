import numpy as np

import environments.cart_pole.environment as cart_pole

def anti_curriculum_learning(min_length=0.25, max_length=0.5, total_evaluations=100):
    lengths = np.linspace(max_length, min_length, total_evaluations)

    def curriculum(env, evaluation, total_evaluations, **metrics):
        index = min(evaluation, total_evaluations - 1)
        new_length = lengths[index]
        cart_pole.update_env_parameters(env, length=new_length)
    
    return curriculum