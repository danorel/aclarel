import environments.mountain_car.environment as mountain_car 

def teacher_student_curriculum(min_gravity=0.0025, max_gravity=0.025, min_reward = -200, max_reward = -110):
    def curriculum(env, evaluation, total_evaluations, **metrics):
        current_gravity = env.unwrapped.gravity

        mean_reward = metrics.get('mean_reward', 0.0)

        reward_ratio = (mean_reward - min_reward) / (max_reward - min_reward)
        target_gravity = min_gravity + reward_ratio * (max_gravity - min_gravity)

        adjustment_speed = (max_gravity - min_gravity) / total_evaluations
        if target_gravity > current_gravity:
            new_gravity = min(target_gravity, current_gravity + adjustment_speed)
        else:
            new_gravity = max(target_gravity, current_gravity - adjustment_speed)

        mountain_car.update_env_parameters(env, gravity=new_gravity)
    
    return curriculum