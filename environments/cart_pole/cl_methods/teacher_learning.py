import environments.cart_pole.environment as cart_pole

def teacher_student_curriculum(min_length=0.25, max_length=0.5, min_reward = 0, max_reward = 500):
    def curriculum(env, evaluation, total_evaluations, **metrics):
        current_length = env.unwrapped.length

        mean_reward = metrics.get('mean_reward', 0.0)

        reward_ratio = (mean_reward - min_reward) / (max_reward - min_reward)
        target_length = min_length + reward_ratio * (max_length - min_length)

        adjustment_speed = (max_length - min_length) / total_evaluations
        if target_length > current_length:
            new_length = min(target_length, current_length + adjustment_speed)
        else:
            new_length = max(target_length, current_length - adjustment_speed)

        cart_pole.update_env_parameters(env, length=new_length)
    
    return curriculum