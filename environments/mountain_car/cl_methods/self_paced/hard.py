import environments.mountain_car.environment as mountain_car

def hard(min_gravity=0.00025, max_gravity=0.0025, milestones=[0.25, 0.5, 0.75]):
    def curriculum(env, evaluation, total_evaluations, **metrics):
        progress = sum(1 for m in milestones if evaluation / total_evaluations > m) / len(milestones)
        new_gravity = min_gravity + (max_gravity - min_gravity) * progress
        mountain_car.update_env_parameters(env, length=new_gravity)
    return curriculum