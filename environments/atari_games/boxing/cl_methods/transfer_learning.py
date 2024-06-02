import numpy as np

import environments.atari_games.boxing.environment as boxing

def transfer_learning_curriculum(min_frame_skip=1, max_frame_skip=10, source_frame_skip=1, target_frame_skip=4, source_evaluations=50, target_evaluations=100):
    source_frame_skips = np.round(np.linspace(source_frame_skip, target_frame_skip, source_evaluations)).astype(int)
    target_frame_skips = [target_frame_skip] * (target_evaluations - source_evaluations)

    def curriculum(env, evaluation, total_evaluations, **metrics):
        if evaluation < source_evaluations:
            index = min(evaluation, source_evaluations - 1)
            new_frame_skip = min(max(int(source_frame_skips[index]), min_frame_skip), max_frame_skip)
        else:
            index = min(evaluation - source_evaluations, total_evaluations - source_evaluations - 1)
            new_frame_skip = min(max(int(target_frame_skips[index]), min_frame_skip), max_frame_skip)
        boxing.update_env_parameters(env, frame_skip=new_frame_skip)
    
    return curriculum