import abc
import pathlib
import matplotlib.pyplot as plt

from collections import defaultdict

import environments.cart_pole.environment as cart_pole_env

ROOT_DIR = pathlib.Path().cwd()
AGENT_DIR = ROOT_DIR / 'agents' / 'cart_pole'
MEASUREMENTS_DIR = ROOT_DIR / 'measurements' / 'cart_pole'

class Agent(abc.ABC):
    def __init__(self, agent_name, curriculum_name) -> None:
        super().__init__()
        # Metric collection state
        self.measurements = defaultdict(list)
        # Path which should serve for agents artifacts
        self.parameters = f'lr-{cart_pole_env.alpha}_gamma-{cart_pole_env.gamma}'
        self.model_dir = AGENT_DIR / curriculum_name / agent_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.measurements_dir = MEASUREMENTS_DIR / curriculum_name / agent_name
        self.measurements_dir.mkdir(parents=True, exist_ok=True)
    
    @abc.abstractmethod
    def act(self, state):
        pass

    @abc.abstractmethod
    def train(self, prev_state, action, reward, next_state):
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

    def track_measurements(self, aar, ses, stability, performance):
        self.measurements['aar'].append(aar)
        self.measurements['ses'].append(ses)
        self.measurements['stability'].append(stability)
        self.measurements['performance'].append(performance)

    def plot_measurements(self):
        for metric, measurements in self.measurements.items():
            measurements_path = self.measurements_dir / f'{metric}_{self.parameters}.png'
            plt.figure() # Create a new figure to prevent overlapping plots
            plt.plot(measurements, label=metric)
            plt.title(f'{metric.capitalize()} over Time')
            plt.xlabel('Episodes')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)
            plt.savefig(measurements_path)
            plt.close()