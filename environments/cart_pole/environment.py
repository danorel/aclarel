import gym
import numpy as np

from tqdm import tqdm

from .rl_methods import Agent

USE_RENDER = False
USE_PRETRAINED_MODEL = False

# Initialize environment
env = gym.make('CartPole-v1', render_mode='human' if USE_RENDER else 'rgb_array')

# Define parameters
alpha = 0.05
gamma = 0.99
initial_epsilon = 0.5  # Start with a high epsilon
epsilon_decay = 0.999  # Decay factor for epsilon
minimum_epsilon = 0.005  # Minimum value for epsilon
total_episodes = 1000
print_interval = 250
evaluation_interval = 10
render_interval = 500

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

# Metric AAR
def calculate_aar(rewards, lengths, adjustment_factors):
    # Adjust rewards by episode length and difficulty adjustment factor
    adjusted_rewards = [(reward / length) * factor for reward, length, factor in zip(rewards, lengths, adjustment_factors)]
    # Return the average of these adjusted rewards
    return np.mean(adjusted_rewards)

# Metric SES
def calculate_ses(actions_taken, terminal_states):
    safe_actions = sum(1 for (action, is_exploratory), terminal in zip(actions_taken, terminal_states) if not terminal and is_exploratory)
    exploration_actions = sum(1 for action, is_exploratory in actions_taken if is_exploratory)
    return safe_actions / exploration_actions if exploration_actions > 0 else 0

def curriculum_adjustment_factor(env, episode):
    # Constants for adjustment calculations
    default_pole_length = 0.5  # Default pole length in meters
    min_pole_length = 0.25     # Minimum pole length considered in curriculum

    current_pole_length = env.unwrapped.length

    # Calculate a baseline adjustment based on pole length
    if current_pole_length <= default_pole_length:
        length_adjustment = default_pole_length / current_pole_length
    else:
        length_adjustment = 1

    # Introduce an episode-dependent adjustment that gradually increases difficulty
    # This could be a linear progression or any other function of the episode number
    episode_factor = 1 + (episode / total_episodes) * (min_pole_length / current_pole_length - 1)

    # Combine the two adjustments
    adjustment_factor = length_adjustment * episode_factor
    
    return adjustment_factor

# A function to evaluate RL agent
def evaluate_agent(agent: Agent, render = False):
    total_reward = 0

    for _ in range(evaluation_interval):
        state, _ = env.reset()
        state = get_discrete_state(state)
        done = False
        while not done:
            if render:
                env.render()  # Render the environment to visualize the agent's performance
            action = agent.act(state)
            state, reward, done, _, _ = env.step(action)
            state = get_discrete_state(state)
            total_reward += reward

    return total_reward / evaluation_interval

# A function to train RL agent
def train_agent(agent: Agent, epsilon, curriculum):
    actions_taken = []
    adjustment_factors = []
    terminal_states = []
    episode_rewards = []
    episode_lengths = []

    for episode in range(evaluation_interval):
        if curriculum is not None:
            curriculum(env, episode, total_episodes)

        state, _ = env.reset()
        state = get_discrete_state(state)
        done = False
        total_reward = 0
        length = 0

        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
                is_exploratory = True
            else:
                action = agent.act(state)
                is_exploratory = False

            next_state, reward, done, _, _ = env.step(action)
            next_state = get_discrete_state(next_state)
            agent.train(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            length += 1

            adjustment_factor = curriculum_adjustment_factor(env, episode)
            adjustment_factors.append(adjustment_factor)
            actions_taken.append((action, is_exploratory))
            terminal_states.append(done)

        episode_rewards.append(total_reward)
        episode_lengths.append(length)
        epsilon = max(minimum_epsilon, epsilon * epsilon_decay)

    stability = np.std(episode_rewards)
    aar = calculate_aar(episode_rewards, episode_lengths, adjustment_factors)
    ses = calculate_ses(actions_taken, terminal_states)

    return agent, epsilon, aar, ses, stability

# A function to both train and evaluate RL agent
def train_evaluate(agent: Agent, curriculum):
    epsilon = initial_epsilon

    for evaluation in tqdm(range(total_episodes // evaluation_interval + 1)):
        render = USE_RENDER and evaluation % render_interval == 0
        agent, epsilon, aar, ses, stability = train_agent(agent, epsilon, curriculum)
        performance = evaluate_agent(agent, render)
        if evaluation % print_interval == 0:
            print(f"AAR: {aar}, SES: {ses}, Learning Stability: {stability}")
            print(f"Evaluation: {evaluation}, Epsilon: {epsilon}, Performance: {performance}")
        agent.track_measurements(aar, ses, stability, performance)

    agent.plot_measurements()
    agent.serialize_agent()
    env.close()