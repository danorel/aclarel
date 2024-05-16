import environments.cart_pole.environment as cart_pole

def one_pass(min_length=0.25, max_length=0.5, total_evaluations=100):
    lengths = [min_length + (max_length - min_length) * i / (total_evaluations - 1) for i in range(total_evaluations)]

    def curriculum(env, evaluation, total_evaluations, **metrics):
        index = min(evaluation, total_evaluations - 1) # ensure the episode number doesn't exceed predefined lengths
        new_length = lengths[index]
        cart_pole.update_env_parameters(env, length=new_length)
    
    return curriculum