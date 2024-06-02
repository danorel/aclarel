import environments.atari_games.boxing.environment as boxing

def hard(min_frame_skip=1, max_frame_skip=10, milestones=[0.25, 0.5, 0.75]):
    def curriculum(env, evaluation, total_evaluations, **metrics):
        progress = sum(1 for m in milestones if evaluation / total_evaluations > m) / len(milestones)
        new_frame_skip = max(int(min_frame_skip + (max_frame_skip - min_frame_skip) * progress), 1)
        boxing.update_env_parameters(env, frame_skip=new_frame_skip)
    return curriculum