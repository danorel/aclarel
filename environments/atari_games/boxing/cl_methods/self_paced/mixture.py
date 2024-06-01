import environments.atari_games.boxing.environment as boxing

def mixture(min_frame_skip=1, max_frame_skip=10, switch=0.5, exponent=2):
    def curriculum(env, evaluation, total_evaluations, **metrics):
        if evaluation / total_evaluations < switch:
            progress = evaluation / (total_evaluations * switch)
        else:
            progress = ((evaluation - total_evaluations * switch) / (total_evaluations * (1 - switch))) ** exponent
        new_frame_skip = min_frame_skip + (max_frame_skip - min_frame_skip) * progress
        boxing.update_env_parameters(env, frame_skip=new_frame_skip)
    return curriculum