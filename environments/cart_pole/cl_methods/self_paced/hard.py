import environments.cart_pole.environment as cart_pole

def hard(min_length=0.5, max_length=0.5, milestones=[0.25, 0.5, 0.75]):
    def curriculum(env, evaluation, total_evaluations, **metrics):
        progress = sum(1 for m in milestones if evaluation / total_evaluations > m) / len(milestones)
        new_length = min_length + (max_length - min_length) * progress
        cart_pole.update_env_parameters(env, length=new_length)
    return curriculum