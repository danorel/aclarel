import gym
import numpy as np

from tqdm import tqdm

import environments.cart_pole.metrics as metrics
import environments.cart_pole.rl_methods as rl_methods

# Initialize environment
env = gym.make('CartPole-v1', render_mode='rgb_array')

# Discretize the observation space
amount_of_bins = 20
state_bins = [
    np.linspace(-4.8, 4.8, amount_of_bins),     # Position
    np.linspace(-4, 4, amount_of_bins),         # Velocity
    np.linspace(-0.418, 0.418, amount_of_bins), # Angle
    np.linspace(-4, 4, amount_of_bins)          # Angular velocity
]

# Model dimensions
state_size = 4
action_size = env.action_space.n

def update_env_parameters(env, length=None, masscart=None, masspole=None, force_mag=None, gravity=None):
    if length is not None:
        env.unwrapped.length = length  # Change pole length
    if masscart is not None:
        env.unwrapped.total_mass = masscart + env.unwrapped.masspole  # Update total mass of the cart
    if masspole is not None:
        env.unwrapped.masspole = masspole  # Change pole mass
        env.unwrapped.total_mass = env.unwrapped.masscart + masspole  # Update total mass
    if force_mag is not None:
        env.unwrapped.force_mag = force_mag  # Change the force magnitude applied
    if gravity is not None:
        env.unwrapped.gravity = gravity  # Adjust gravity

# A function to convert continuous states to discrete indices
def get_discrete_state(state):
    index = []
    for i, val in enumerate(state):
        index.append(np.digitize(val, state_bins[i]) - 1)  # Find the bin index for each state dimension
    return tuple(index)

# A function to evaluate RL agent
def evaluate_agent(agent: rl_methods.Agent, use_render: bool = False):
    rewards_per_episode = np.array([])
    successful_episodes = 0

    evaluation_interval = agent.hyperparameters['evaluation_interval']
    max_episode_length = env.spec.max_episode_steps

    for _ in range(evaluation_interval):
        reward_per_episode = 0
        step_count = 0
        state, _ = env.reset()
        done, truncated = False, False
        while not done and not truncated:
            if use_render:
                env.render()  # Render the environment to visualize the agent's performance
            state = get_discrete_state(state)
            action, _ = agent.act(state, greedily=True)
            state, reward, done, truncated, _ = env.step(action)
            reward_per_episode += reward
            step_count += 1
        rewards_per_episode = np.append(rewards_per_episode, reward_per_episode)
        if done and not truncated and step_count >= max_episode_length:
            successful_episodes += 1

    mean_reward = rewards_per_episode.mean()
    std_reward = rewards_per_episode.std()
    total_reward = rewards_per_episode.sum()
    success_rate = successful_episodes / evaluation_interval

    return mean_reward, std_reward, total_reward, success_rate

# A function to train RL agent
def train_agent(agent: rl_methods.Agent, epsilon: float, curriculum):
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
                episode, 
                total_episodes, 
                metrics={ 'aar': metrics.aar(episode_rewards, episode_lengths, adjustment_factors) }
            )

        state, _ = env.reset()
        state = get_discrete_state(state)
        done, truncated = False, False
        total_reward = 0
        length = 0

        while not done and not truncated:
            action, is_exploratory = agent.act(state, greedily=False)
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = get_discrete_state(next_state)
            agent.train(state, action, reward, next_state, done)
            state = next_state

            total_reward += reward
            length += 1

            adjustment_factor = metrics.adjustment_factor(env, episode, total_episodes)
            adjustment_factors.append(adjustment_factor)
            actions_taken.append((action, is_exploratory))
            terminal_states.append(done)

        episode_rewards.append(total_reward)
        episode_lengths.append(length)
        epsilon = max(minimum_epsilon, epsilon * epsilon_decay)

    stability = np.std(episode_rewards)
    aar = metrics.aar(episode_rewards, episode_lengths, adjustment_factors)
    ses = metrics.ses(actions_taken, terminal_states)

    return agent, epsilon, aar, ses, stability

# A function to both train and evaluate RL agent
def train_evaluate(agent: rl_methods.Agent, curriculum, use_render: bool = False):
    env = gym.make('CartPole-v1', render_mode='human' if use_render else 'rgb_array')

    total_episodes = agent.hyperparameters['total_episodes']
    initial_epsilon = agent.hyperparameters['initial_epsilon']
    evaluation_interval = agent.hyperparameters['evaluation_interval']
    print_interval = agent.hyperparameters['print_interval']

    epsilon = initial_epsilon
    for evaluation in tqdm(range(total_episodes // evaluation_interval + 1)):
        agent, epsilon, aar, ses, learning_stability = train_agent(agent, epsilon, curriculum)
        mean_reward, std_reward, total_reward, success_rate = evaluate_agent(agent, use_render)
        if evaluation % print_interval == 0:
            print(f"Evaluation {evaluation} (Epsilon={epsilon}):")
            print(f"\tAAR: {aar}\n \tSES: {ses}\n \tLearning Stability: {learning_stability}\n \tMean Reward: {mean_reward}\n \tStd Reward: {std_reward}\n")
        agent.track_measurements(evaluation, aar, ses, learning_stability, mean_reward, std_reward, total_reward, success_rate)

    agent.plot_measurements()
    agent.serialize_agent()
    env.close()