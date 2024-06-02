import numpy as np

import environments.atari_games.boxing.environment as boxing

def logarithmic(min_frame_skip=1, max_frame_skip=10):
    def curriculum(env, evaluation, total_evaluations, **metrics):
        scale = np.log(total_evaluations)
        new_frame_skip = min(max(int(min_frame_skip + (max_frame_skip - min_frame_skip) * (np.log(evaluation + 1) / scale)), min_frame_skip), max_frame_skip)
        boxing.update_env_parameters(env, frame_skip=new_frame_skip)
    return curriculum