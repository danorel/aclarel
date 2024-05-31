import environments.mountain_car.environment as mountain_car 

def mixture(min_gravity=0.0025, max_gravity=0.025, switch_length=0.5, exponent=2):
    def curriculum(env, evaluation, total_evaluations, **metrics):
        if evaluation / total_evaluations < switch_length:
            progress = evaluation / (total_evaluations * switch_length)
        else:
            progress = ((evaluation - total_evaluations * switch_length) / (total_evaluations * (1 - switch_length))) ** exponent
        new_gravity = min_gravity + (max_gravity - min_gravity) * progress
        mountain_car.update_env_parameters(env, gravity=new_gravity)
    return curriculum