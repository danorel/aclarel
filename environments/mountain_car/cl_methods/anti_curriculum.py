import numpy as np

import environments.mountain_car.environment as mountain_car 

def anti_curriculum_learning(min_gravity=0.00025, max_gravity=0.0025, total_evaluations=100):
    gravities = np.linspace(max_gravity, min_gravity, total_evaluations)

    def curriculum(env, evaluation, total_evaluations, **metrics):
        index = min(evaluation, total_evaluations - 1)
        new_gravity = gravities[index]
        mountain_car.update_env_parameters(env, gravity=new_gravity)
    
    return curriculum