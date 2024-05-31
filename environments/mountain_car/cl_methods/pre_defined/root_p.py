import numpy as np

import environments.mountain_car.environment as mountain_car

def root_p(min_gravity=0.00025, max_gravity=0.0025, exponent=2):
    def curriculum(env, evaluation, total_evaluations, **metrics):
        evaluation = max(evaluation, 1)
        sqrt_evaluation = np.sqrt(evaluation)
        new_gravity = min_gravity + (max_gravity - min_gravity) * sqrt_evaluation / (exponent * sqrt_evaluation)
        mountain_car.update_env_parameters(env, gravity=new_gravity)
    return curriculum