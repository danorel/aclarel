import environments.mountain_car.environment as mountain_car 

def linear(min_gravity=0.00025, max_gravity=0.0025):
    def curriculum(env, evaluation, total_evaluations, **metrics):
        new_gravity = min_gravity + (max_gravity - min_gravity) * evaluation / total_evaluations
        mountain_car.update_env_parameters(env, gravity=new_gravity)
    return curriculum