import environments.cart_pole.environment as cart_pole

def mixture(base_length=0.5, max_length=0.5, switch_point=0.5, exponent=2):
    def curriculum(env, episode, total_episodes, **metrics):
        if episode / total_episodes < switch_point:
            progress = episode / (total_episodes * switch_point) # linear phase
        else:
            progress = ((episode - total_episodes * switch_point) / (total_episodes * (1 - switch_point))) ** exponent # polynomial phase
        new_length = base_length + (max_length - base_length) * progress
        cart_pole.update_env_parameters(env, length=new_length)
    return curriculum