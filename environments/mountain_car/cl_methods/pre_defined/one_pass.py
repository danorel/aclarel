import environments.mountain_car.environment as mountain_car

def one_pass(min_gravity=0.0025, max_gravity=0.025, total_evaluations=100):
    gravities = [min_gravity + (max_gravity - min_gravity) * i / (total_evaluations - 1) for i in range(total_evaluations)]

    def curriculum(env, evaluation, total_evaluations, **metrics):
        index = min(evaluation, total_evaluations - 1)
        new_gravity = gravities[index]
        mountain_car.update_env_parameters(env, gravity=new_gravity)
    
    return curriculum