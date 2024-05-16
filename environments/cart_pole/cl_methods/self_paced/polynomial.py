import environments.cart_pole.environment as cart_pole

def polynomial(min_length=0.25, max_length=0.5, exponent=2):
    def curriculum(env, evaluation, total_evaluations, **metrics):
        progress = (evaluation / total_evaluations) ** exponent
        new_length = min_length + (max_length - min_length) * progress
        cart_pole.update_env_parameters(env, length=new_length)
    return curriculum