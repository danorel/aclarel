import environments.cart_pole.environment as cart_pole

def hard(base_length=0.5, max_length=0.5, milestones=[0.25, 0.5, 0.75]):
    def curriculum(env, episode, total_episodes):
        progress = sum(1 for m in milestones if episode / total_episodes > m) / len(milestones)
        new_length = base_length + (max_length - base_length) * progress
        cart_pole.update_env_parameters(env, length=new_length)
    return curriculum