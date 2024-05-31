import numpy as np

import environments.mountain_car.environment as mountain_car 

def transfer_learning_curriculum(source_gravity=0.00025, target_gravity=0.0025, source_evaluations=50, target_evaluations=100):
    source_gravities = np.linspace(source_gravity, target_gravity, source_evaluations)
    target_gravities = [target_gravity] * (target_evaluations - source_evaluations)

    def curriculum(env, evaluation, total_evaluations, **metrics):
        if evaluation < source_evaluations:
            index = min(evaluation, source_evaluations - 1)
            new_gravity = source_gravities[index]
        else:
            index = min(evaluation - source_evaluations, total_evaluations - source_evaluations - 1)
            new_gravity = target_gravities[index]
        mountain_car.update_env_parameters(env, gravity=new_gravity)
    
    return curriculum