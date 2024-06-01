import numpy as np

import environments.atari_games.boxing.environment as boxing

def root_p(min_frame_skip=1, max_frame_skip=10, exponent=2):
    def curriculum(env, evaluation, total_evaluations, **metrics):
        evaluation = max(evaluation, 1)
        sqrt_evaluation = np.sqrt(evaluation)
        new_frame_skip = min_frame_skip + (max_frame_skip - min_frame_skip) * sqrt_evaluation / (exponent * sqrt_evaluation)
        boxing.update_env_parameters(env, frame_skip=new_frame_skip)
    return curriculum