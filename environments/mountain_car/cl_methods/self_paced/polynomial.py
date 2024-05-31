import environments.mountain_car.environment as mountain_car 

def polynomial(min_gravity=0.0025, max_gravity=0.025, exponent=2):
    def curriculum(env, evaluation, total_evaluations, **metrics):
        progress = (evaluation / total_evaluations) ** exponent
        new_gravity = min_gravity + (max_gravity - min_gravity) * progress
        mountain_car.update_env_parameters(env, length=new_gravity)
    return curriculum