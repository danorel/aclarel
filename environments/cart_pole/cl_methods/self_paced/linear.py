import environments.cart_pole.environment as cart_pole

def linear(min_length=0.25, max_length=0.5):
    def curriculum(env, evaluation, total_evaluations, **metrics):
        new_length = min_length + (max_length - min_length) * evaluation / total_evaluations
        cart_pole.update_env_parameters(env, length=new_length)
    return curriculum