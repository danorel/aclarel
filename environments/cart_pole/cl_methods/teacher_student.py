import numpy as np

import environments.cart_pole.metrics as metrics 
import environments.cart_pole.environment as cart_pole

def teacher_student_curriculum(base_length=0.5, max_length=1.0, total_episodes=100):
    lengths = np.linspace(base_length, max_length, total_episodes)

    def curriculum(env, episode, total_episodes, **metrics):
        aar = metrics.get('aar', 0) # performance
        lengths_sorted = sorted(lengths, key=lambda length: abs(length - aar))
        new_length = lengths_sorted[0]
        cart_pole.update_env_parameters(env, length=new_length)
    
    return curriculum