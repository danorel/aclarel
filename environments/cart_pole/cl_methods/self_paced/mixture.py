import environments.cart_pole.environment as cart_pole

def mixture(min_length=0.25, max_length=0.5, switch_length=0.5, exponent=2):
    def curriculum(env, evaluation, total_evaluations, **metrics):
        if evaluation / total_evaluations < switch_length:
            progress = evaluation / (total_evaluations * switch_length) # linear phase
        else:
            progress = ((evaluation - total_evaluations * switch_length) / (total_evaluations * (1 - switch_length))) ** exponent # polynomial phase
        new_length = min_length + (max_length - min_length) * progress
        cart_pole.update_env_parameters(env, length=new_length)
    return curriculum