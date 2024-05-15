import environments.cart_pole.environment as cart_pole

def one_pass(base_length=0.5, max_length=1.0, total_episodes=100):
    lengths = [base_length + (max_length - base_length) * i / (total_episodes - 1) for i in range(total_episodes)]

    def curriculum(env, episode, total_episodes, **metrics):
        index = min(episode, total_episodes - 1) # ensure the episode number doesn't exceed predefined lengths
        new_length = lengths[index]
        cart_pole.update_env_parameters(env, length=new_length)
    
    return curriculum