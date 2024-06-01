import environments.atari_games.boxing.environment as boxing 

def one_pass(min_frame_skip=1, max_frame_skip=10, total_evaluations=100):
    frame_skips = [min_frame_skip + (max_frame_skip - min_frame_skip) * i / (total_evaluations - 1) for i in range(total_evaluations)]

    def curriculum(env, evaluation, total_evaluations, **metrics):
        index = min(evaluation, total_evaluations - 1)
        new_frame_skip = frame_skips[index]
        boxing.update_env_parameters(env, frame_skip=new_frame_skip)
    
    return curriculum