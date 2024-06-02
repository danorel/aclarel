import environments.atari_games.boxing.environment as boxing

def teacher_student_curriculum(min_frame_skip=1, max_frame_skip=10, min_reward = -100, max_reward = 100):
    def curriculum(env, evaluation, total_evaluations, **metrics):
        current_frame_skip = env.unwrapped._frameskip

        mean_reward = metrics.get('mean_reward', 0.0)

        reward_ratio = (mean_reward - min_reward) / (max_reward - min_reward)
        target_frame_skip = int(min_frame_skip + reward_ratio * (max_frame_skip - min_frame_skip))

        adjustment_speed = (max_frame_skip - min_frame_skip) / total_evaluations
        if target_frame_skip > current_frame_skip:
            new_frame_skip = min(max(int(min(target_frame_skip, current_frame_skip + adjustment_speed)), min_frame_skip), max_frame_skip)
        else:
            new_frame_skip = min(max(int(max(target_frame_skip, current_frame_skip - adjustment_speed)), min_frame_skip), max_frame_skip)

        boxing.update_env_parameters(env, frame_skip=new_frame_skip)
    
    return curriculum