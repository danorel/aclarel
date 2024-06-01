import numpy as np

import environments.atari_games.boxing.environment as boxing

def logistic(min_frame_skip=1, max_frame_skip=10, growth_rate=10):
    def curriculum(env, evaluation, total_evaluations, **metrics):
        evaluation = max(evaluation, 1)
        progress = (evaluation / total_evaluations) * growth_rate
        sigmoid = 1 / (1 + np.exp(-progress + growth_rate / 2))
        new_frame_skip = min_frame_skip + (max_frame_skip - min_frame_skip) * sigmoid
        boxing.update_env_parameters(env, frame_skip=new_frame_skip)
    return curriculum