import numpy as np

import environments.mountain_car.environment as mountain_car 

def logistic(min_gravity=0.0025, max_gravity=0.025, growth_rate=10):
    def curriculum(env, evaluation, total_evaluations, **metrics):
        evaluation = max(evaluation, 1) # avoid division by zero
        progress = (evaluation / total_evaluations) * growth_rate
        sigmoid = 1 / (1 + np.exp(-progress + growth_rate / 2))
        new_gravity = min_gravity + (max_gravity - min_gravity) * sigmoid
        mountain_car.update_env_parameters(env, gravity=new_gravity)
    return curriculum