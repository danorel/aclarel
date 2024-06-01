import numpy as np

import environments.atari_games.boxing.environment as boxing

def anti_curriculum_learning(min_frame_skip=1, max_frame_skip=10, total_evaluations=100):
    frame_skips = np.linspace(max_frame_skip, min_frame_skip, total_evaluations)

    def curriculum(env, evaluation, total_evaluations, **metrics):
        index = min(evaluation, total_evaluations - 1)
        new_frame_skip = frame_skips[index]
        boxing.update_env_parameters(env, frame_skip=new_frame_skip)
    
    return curriculum