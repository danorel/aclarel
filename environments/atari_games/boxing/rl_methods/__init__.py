import abc
import pandas as pd
import pathlib
import matplotlib.pyplot as plt

ROOT_DIR = pathlib.Path().cwd()
AGENT_DIR = ROOT_DIR / 'agents' / 'atari_games' / 'boxing'
MEASUREMENTS_DIR = ROOT_DIR / 'measurements' / 'atari_games' / 'boxing'

class Agent(abc.ABC):
    def __init__(self, agent_name, curriculum_name) -> None:
        super().__init__()
        # Agent metadata and metric collection state
        self.metadata = {
            "agent_name": agent_name,
            "curriculum_name": curriculum_name
        }
        self.hyperparameters: dict = None
        self.hyperparameter_path: str = None
        self.measurements = pd.DataFrame(columns=['agent_name', 'evaluation', 'curriculum_name', 'aar', 'ses', 'learning_stability', 'mean_reward', 'std_reward', 'total_reward', 'success_rate'])
        # Path which should serve for agents artifacts
        self.model_dir = AGENT_DIR / curriculum_name / agent_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.measurements_dir = MEASUREMENTS_DIR / curriculum_name / agent_name
        self.measurements_dir.mkdir(parents=True, exist_ok=True)
    
    @abc.abstractmethod
    def act(self, state, greedily: bool = False):
        pass

    @abc.abstractmethod
    def train(self, prev_state, action, reward, next_state, done, log_prob):
        pass

    @abc.abstractmethod
    def refresh_agent(self):
        pass

    @abc.abstractmethod
    def deserialize_agent(self):
        pass

    @abc.abstractmethod
    def serialize_agent(self):
        pass

    def track_measurements(self, evaluation, aar, ses, learning_stability, mean_reward, std_reward, total_reward, success_rate):
        measurement = {
            **self.metadata,
            'evaluation': evaluation,
            'aar': aar,
            'ses': ses,
            'learning_stability': learning_stability,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'total_reward': total_reward,
            'success_rate': success_rate
        }
        self.measurements = pd.concat([self.measurements, pd.DataFrame([measurement])], ignore_index=True)

    def plot_measurements(self):
        for metric in self.measurements.columns.difference([*self.metadata.keys(), 'evaluation']):
            measurements_path = self.measurements_dir / f'{metric}_{self.hyperparameter_path}.png'
            plt.figure() # Create a new figure to prevent overlapping plots
            plt.plot(self.measurements.index, self.measurements[metric], label=metric)
            plt.title(f'{metric.capitalize()} over Time')
            plt.xlabel('Episodes')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)
            plt.savefig(measurements_path)
            plt.close()