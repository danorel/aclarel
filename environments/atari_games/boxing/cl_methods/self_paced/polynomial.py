import environments.atari_games.boxing.environment as boxing

def polynomial(min_frame_skip=1, max_frame_skip=10, exponent=2):
    def curriculum(env, evaluation, total_evaluations, **metrics):
        progress = (evaluation / total_evaluations) ** exponent
        new_frame_skip = min_frame_skip + (max_frame_skip - min_frame_skip) * progress
        boxing.update_env_parameters(env, frame_skip=new_frame_skip)
    return curriculum