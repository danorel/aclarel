import numpy as np

import environments.mountain_car.environment as mountain_car 

def logarithmic(min_gravity=0.0025, max_gravity=0.025):
    def curriculum(env, evaluation, total_evaluations, **metrics):
        scale = np.log(total_evaluations)
        new_gravity = min_gravity + (max_gravity - min_gravity) * (np.log(evaluation + 1) / scale)
        mountain_car.update_env_parameters(env, gravity=new_gravity)
    return curriculum