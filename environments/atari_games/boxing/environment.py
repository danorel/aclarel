import gym
import numpy as np

from tqdm import tqdm

import environments.atari_games.boxing.metrics as metrics
import environments.atari_games.boxing.rl_methods as rl_methods

num_envs = 4
env = gym.make('ALE/Boxing-v5', render_mode='rgb_array', frameskip=4)

reward_threshold = 100
action_size = env.action_space.n

def update_env_parameters(env, frame_skip=None):
    if frame_skip is not None:
        env.unwrapped._frameskip = frame_skip

def evaluate_agent(env, agent: rl_methods.Agent, use_render: bool = False):
    rewards_per_episode = np.array([])
    successful_episodes = 0

    evaluation_interval = agent.hyperparameters['evaluation_interval']

    for _ in range(evaluation_interval):
        reward_per_episode = 0
        step_count = 0
        state, _ = env.reset()
        done, truncated = False, False
        while not done and not truncated:
            if use_render:
                env.render()
            action, _ = agent.act(state)
            state, reward, done, truncated, _ = env.step(action)
            reward_per_episode += reward
            step_count += 1
        rewards_per_episode = np.append(rewards_per_episode, reward_per_episode)
        if reward_per_episode >= reward_threshold:
            successful_episodes += 1

    mean_reward = rewards_per_episode.mean()
    std_reward = rewards_per_episode.std()
    total_reward = rewards_per_episode.sum()
    success_rate = successful_episodes / evaluation_interval

    return mean_reward, std_reward, total_reward, success_rate

def train_agent(env, agent: rl_methods.Agent, epsilon: float, curriculum, evaluation, total_evaluations):
    actions_taken = []
    adjustment_factors = []
    terminal_states = []
    episode_rewards = []
    episode_lengths = []

    total_episodes = agent.hyperparameters['total_episodes']
    evaluation_interval = agent.hyperparameters['evaluation_interval']
    minimum_epsilon = agent.hyperparameters['minimum_epsilon']
    epsilon_decay = agent.hyperparameters['epsilon_decay']

    for episode in range(evaluation_interval):
        if curriculum is not None:
            curriculum(
                env,
                evaluation,
                total_evaluations, 
                **{ 
                    'aar': metrics.aar(episode_rewards, episode_lengths, adjustment_factors),
                    'mean_reward': np.mean(episode_rewards) if len(episode_rewards) else 0.0
                }
            )

        state, _ = env.reset()
        done, truncated = False, False
        total_reward = 0
        length = 0

        while not done and not truncated:
            action, props = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.train(state, action, reward, next_state, done, props['log_prob'])
            state = next_state

            total_reward += reward
            length += 1

            adjustment_factor = metrics.adjustment_factor(env, episode, total_episodes)
            adjustment_factors.append(adjustment_factor)
            actions_taken.append((action, props['is_exploratory']))
            terminal_states.append(done)

        episode_rewards.append(total_reward)
        episode_lengths.append(length)
        epsilon = max(minimum_epsilon, epsilon * epsilon_decay)

    stability = np.std(episode_rewards)
    aar = metrics.aar(episode_rewards, episode_lengths, adjustment_factors)
    ses = metrics.ses(actions_taken, terminal_states)

    return agent, epsilon, aar, ses, stability

def train_evaluate(agent: rl_methods.Agent, curriculum, use_render: bool = False):
    training_env = gym.make('ALE/Boxing-v5', render_mode='rgb_array', frameskip=4)
    evaluation_env = gym.make('ALE/Boxing-v5', render_mode='human' if use_render else 'rgb_array', frameskip=4)

    total_episodes = agent.hyperparameters['total_episodes']
    initial_epsilon = agent.hyperparameters['initial_epsilon']
    evaluation_interval = agent.hyperparameters['evaluation_interval']
    print_interval = agent.hyperparameters['print_interval']

    total_evaluations = total_episodes // evaluation_interval
    epsilon = initial_epsilon
    for evaluation in tqdm(range(total_evaluations + 1)):
        agent, epsilon, aar, ses, learning_stability = train_agent(training_env, agent, epsilon, curriculum, evaluation, total_evaluations)
        mean_reward, std_reward, total_reward, success_rate = evaluate_agent(evaluation_env, agent, use_render)
        if evaluation % print_interval == 0:
            print(f"Evaluation {evaluation} (Epsilon={round(epsilon, 5)}):")
            print(f"\tTraining Frameskip: {round(training_env.unwrapped._frameskip, 3)}\n \tTraining Stability: {round(learning_stability, 3)}\n \tAAR: {round(aar, 3)}\n \tSES: {round(ses, 3)}\n \tMean Reward: {round(mean_reward, 3)}\n \tStd Reward: {round(std_reward, 3)}\n")
            agent.serialize_agent()
        agent.track_measurements(evaluation, aar, ses, learning_stability, mean_reward, std_reward, total_reward, success_rate)

    agent.plot_measurements()
    agent.serialize_agent()
    env.close()