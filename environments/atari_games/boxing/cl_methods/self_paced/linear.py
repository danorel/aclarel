import environments.atari_games.boxing.environment as boxing

def linear(min_frame_skip=1, max_frame_skip=10):
    def curriculum(env, evaluation, total_evaluations, **metrics):
        new_frame_skip = min_frame_skip + (max_frame_skip - min_frame_skip) * evaluation / total_evaluations
        boxing.update_env_parameters(env, frame_skip=new_frame_skip)
    return curriculum