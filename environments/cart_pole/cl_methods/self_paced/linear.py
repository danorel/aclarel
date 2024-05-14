import environments.cart_pole.environment as cart_pole

def linear(base_length=0.5, max_length=0.5):
    def curriculum(env, episode, total_episodes):
        new_length = base_length + (max_length - base_length) * episode / total_episodes
        cart_pole.update_env_parameters(env, length=new_length)
    return curriculum