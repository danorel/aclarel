import abc
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

ROOT_DIR = pathlib.Path().cwd()
AGENT_DIR = ROOT_DIR / 'agents' / 'cart_pole'
MEASUREMENTS_DIR = ROOT_DIR / 'measurements' / 'cart_pole'

class Agent(abc.ABC):
    def __init__(self, agent_name, curriculum_name) -> None:
        super().__init__()
        # Agent metadata and metric collection state
        self.writer = SummaryWriter()
        self.metadata = {
            "agent_name": agent_name,
            "curriculum_name": curriculum_name
        }
        self.hyperparameters: dict = None
        self.hyperparameter_path: str = None
        self.measurements = pd.DataFrame(columns=['agent_name', 'curriculum_name', 'evaluation', 'aar', 'ses', 'learning_stability', 'mean_reward', 'std_reward', 'total_reward', 'success_rate'])
        # Path which should serve for agents artifacts
        self.model_dir = AGENT_DIR / curriculum_name / agent_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.measurements_dir = MEASUREMENTS_DIR / curriculum_name / agent_name
        self.measurements_dir.mkdir(parents=True, exist_ok=True)
        # Tracking steps over time
        self.steps_count = 0
    
    @abc.abstractmethod
    def act(self, state, greedily: bool = False):
        pass

    @abc.abstractmethod
    def train(self, prev_state, action, reward, next_state, done):
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
        if self.steps_count % self.hyperparameters["log_interval"] == 0:
            measurement = pd.DataFrame([{
                **self.metadata,
                'evaluation': evaluation,
                'aar': aar,
                'ses': ses,
                'learning_stability': learning_stability,
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'total_reward': total_reward,
                'success_rate': success_rate
            }])
            if not measurement.isna().any().any():
                self.measurements = pd.concat([self.measurements, measurement], ignore_index=True)
                self.writer.add_scalar('Performance/Mean_Reward', mean_reward, self.steps_count)
                self.writer.add_scalar('Performance/Std_Reward', std_reward, self.steps_count)
                self.writer.add_scalar('Performance/Total_Reward', total_reward, self.steps_count)
                self.writer.add_scalar('Performance/Success_Rate', success_rate, self.steps_count)
                self.writer.add_scalar('Performance/AAR', aar, self.steps_count)
                self.writer.add_scalar('Performance/SES', ses, self.steps_count)

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

    def close(self):
        self.writer.close()