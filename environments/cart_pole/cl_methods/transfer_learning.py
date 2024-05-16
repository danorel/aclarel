import numpy as np

import environments.cart_pole.environment as cart_pole

def transfer_learning_curriculum(source_length=0.25, target_length=0.5, source_evaluations=50, target_evaluations=100):
    source_lengths = np.linspace(source_length, target_length, source_evaluations)
    target_lengths = [target_length] * (target_evaluations - source_evaluations)

    def curriculum(env, evaluation, total_evaluations, **metrics):
        if evaluation < source_evaluations:
            index = min(evaluation, source_evaluations - 1)
            new_length = source_lengths[index]
        else:
            index = min(evaluation - source_evaluations, total_evaluations - source_evaluations - 1)
            new_length = target_lengths[index]
        cart_pole.update_env_parameters(env, length=new_length)
    
    return curriculum