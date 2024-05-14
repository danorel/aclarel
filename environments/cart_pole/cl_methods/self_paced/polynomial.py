import environments.cart_pole.environment as cart_pole

def polynomial(base_length=0.5, max_length=0.5, exponent=2):
    def curriculum(env, episode, total_episodes):
        progress = (episode / total_episodes) ** exponent
        new_length = base_length + (max_length - base_length) * progress
        cart_pole.update_env_parameters(env, length=new_length)
    return curriculum