import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from environment import SpringSliderEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import os
import json
from datetime import datetime
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class ChallengingSpringSliderEnv(SpringSliderEnv):


    def __init__(self, **kwargs):
        if 'target_tolerance' not in kwargs:
            kwargs['target_tolerance'] = 0.01  # 1cm tolerance

        super().__init__(**kwargs)

        self.max_force = 200.0

        # Better reward weights
        self.reward_weights = {
            'accuracy': 300.0,
            'peak_force': 0.002,
            'peak_velocity': 0.3,
            'time': 0.05,
            'success_bonus': 200.0
        }

    def _calculate_reward(self, position, velocity, episode_ended):
        """Shaped rewards during episode."""
        distance_to_target = abs(position - self.target_distance)
        shaped_reward = -10.0 * distance_to_target
        shaped_reward -= 0.1 * abs(velocity)

        if episode_ended:
            distance_error = abs(self.max_distance_reached - self.target_distance)
            accuracy_reward = -self.reward_weights['accuracy'] * distance_error

            # 1cm tolerance bonus
            if distance_error <= self.target_tolerance:
                accuracy_reward += self.reward_weights['success_bonus']

            force_penalty = -self.reward_weights['peak_force'] * self.peak_force
            velocity_penalty = -self.reward_weights['peak_velocity'] * self.peak_velocity
            time_penalty = -self.reward_weights['time'] * self.current_step

            total_reward = accuracy_reward + force_penalty + velocity_penalty + time_penalty
            return total_reward

        return shaped_reward


gym.register(
    id='ChallengingSpringSlider-v0',
    entry_point='__main__:ChallengingSpringSliderEnv',
    max_episode_steps=700,
)


CONFIG = {
    'algorithm': 'SAC',
    'timesteps_per_protocol': 150000,
    'target_distance': 0.20,

    # 5 KEY PROTOCOLS
    'protocols': {
        # Small increments
        'linear_small': {
            'type': 'linear',
            'start': 50,
            'increment': 40,
            'num_levels': 15,
            'description': 'Linear +40 N/m (50-610 N/m)'
        },

        # Large increments
        'linear_large': {
            'type': 'linear',
            'start': 50,
            'increment': 110,
            'num_levels': 6,
            'description': 'Linear +110 N/m (50-600 N/m, large jumps)'
        },

        # Logarithmic
        'logarithmic': {
            'type': 'logarithmic',
            'start': 50,
            'end': 600,
            'num_levels': 15,
            'description': 'Logarithmic 50-600 N/m (medical range)'
        },

        # Random sampling
        'random_sampling': {
            'type': 'random',
            'min': 50,
            'max': 600,
            'num_levels': 15,
            'seed': 42,
            'description': 'Random stiffnesses 50-600 N/m'
        },

        # Patient categories
        'patient_categories': {
            'type': 'custom',
            'levels': [
                # Easy anatomical stiffness
                50, 75, 100, 125, 150,
                # Normal anatomical stiffness
                175, 200, 225, 275, 325,
                # Difficult anatomical stiffness
                350, 400, 475, 550, 600
            ],
            'description': 'Patient-based: easy, normal, difficult anatomical stiffness'
        },
    },

    # Test stiffnesses (14 unseen values)
    'eval_episodes_per_stiffness': 30,
    'test_stiffnesses': [
        # Easy patient range
        60, 85, 110, 135,
        # Normal patient range
        165, 190, 235, 265, 310,
        # Difficult patient range
        360, 425, 490, 540, 580
    ],

    'results_dir': 'results_challenging',
    'models_dir': 'models_challenging',
    'plots_dir': 'plots_challenging',
}


class ProtocolManager:
    @staticmethod
    def generate_protocol(protocol_config):
        ptype = protocol_config['type']

        if ptype == 'linear':
            start = protocol_config['start']
            increment = protocol_config['increment']
            num = protocol_config['num_levels']
            return [start + i * increment for i in range(num)]

        elif ptype == 'logarithmic':
            start = protocol_config['start']
            end = protocol_config['end']
            num = protocol_config['num_levels']
            return np.logspace(np.log10(start), np.log10(end), num).tolist()

        elif ptype == 'random':
            min_val = protocol_config['min']
            max_val = protocol_config['max']
            num = protocol_config['num_levels']
            seed = protocol_config.get('seed', 42)
            np.random.seed(seed)
            levels = np.random.uniform(min_val, max_val, num).tolist()
            return sorted(levels)  # Sort for consistency

        elif ptype == 'decreasing':
            start = protocol_config['start']
            decrement = protocol_config['decrement']
            num = protocol_config['num_levels']
            return [start - i * decrement for i in range(num)]

        elif ptype == 'custom':
            return protocol_config['levels']

        else:
            raise ValueError(f"Unknown protocol type: {ptype}")


class ChallengingMetricsCallback(BaseCallback):
    def __init__(self, protocol_name, verbose=1):
        super().__init__(verbose)
        self.protocol_name = protocol_name
        self.episode_count = 0

        self.metrics = {
            'episodes': [],
            'rewards': [],
            'errors': [],
            'peak_forces': [],
            'peak_velocities': [],
            'success': [],
            'stiffnesses': [],
        }

    def _on_step(self):
        if self.locals.get('dones')[0]:
            info = self.locals['infos'][0]

            self.episode_count += 1
            self.metrics['episodes'].append(self.episode_count)
            self.metrics['rewards'].append(self.locals['rewards'][0])
            self.metrics['errors'].append(info['distance_error'] * 100)
            self.metrics['peak_forces'].append(info['peak_force'])
            self.metrics['peak_velocities'].append(info['peak_velocity'])
            self.metrics['success'].append(info['distance_error'] <= 0.01)  # 1cm!
            self.metrics['stiffnesses'].append(info['stiffness'])

            # Print every 50 episodes
            if self.episode_count % 50 == 0:
                recent_errors = self.metrics['errors'][-20:]
                recent_success = self.metrics['success'][-20:]
                recent_rewards = self.metrics['rewards'][-20:]
                print(f"[{self.protocol_name}] Episode {self.episode_count} | "
                      f"Error: {np.mean(recent_errors):.2f}cm | "
                      f"Success: {np.mean(recent_success) * 100:.0f}% | "
                      f"Reward: {np.mean(recent_rewards):.1f}")

        return True


class CurriculumEnv(ChallengingSpringSliderEnv):
    def __init__(self, stiffness_levels, target_distance=0.20, **kwargs):
        self.stiffness_levels = stiffness_levels
        self.current_level_idx = 0
        super().__init__(
            stiffness=stiffness_levels[0],
            target_distance=target_distance,
            **kwargs
        )

    def reset(self, **kwargs):
        self.stiffness = self.stiffness_levels[self.current_level_idx]
        self.current_level_idx = (self.current_level_idx + 1) % len(self.stiffness_levels)
        return super().reset(**kwargs)


def train_protocol(protocol_name, protocol_config, config):
    print("\n" + "=" * 80)
    print(f"TRAINING: {protocol_name}")
    print(f"Description: {protocol_config['description']}")
    print("=" * 80)

    stiffness_levels = ProtocolManager.generate_protocol(protocol_config)
    print(f"Stiffness levels: {[f'{s:.0f}' for s in stiffness_levels]}")
    print(f"1cm tolerance, 700 max steps")

    env = CurriculumEnv(
        stiffness_levels=stiffness_levels,
        target_distance=config['target_distance'],
        gui=False,
        max_steps=700
    )

    model = SAC(
        'MlpPolicy',
        env,
        verbose=0,
        learning_rate=1e-3,
        buffer_size=300000,
        learning_starts=5000,
        batch_size=512,
        tau=0.01,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
    )

    callback = ChallengingMetricsCallback(protocol_name)
    model.learn(
        total_timesteps=config['timesteps_per_protocol'],
        callback=callback,
        progress_bar=True
    )

    model_path = os.path.join(config['models_dir'], f"{protocol_name}.zip")
    os.makedirs(config['models_dir'], exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")

    env.close()

    return model, callback.metrics, stiffness_levels


def evaluate_on_stiffness(model, stiffness, target_distance, num_episodes=30):
    env = ChallengingSpringSliderEnv(
        stiffness=stiffness,
        target_distance=target_distance,
        gui=False,
        max_steps=700
    )

    errors = []
    forces = []
    velocities = []
    successes = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        errors.append(info['distance_error'] * 100)
        forces.append(info['peak_force'])
        velocities.append(info['peak_velocity'])
        successes.append(info['distance_error'] <= 0.01)  # 1cm tolerance

    env.close()

    return {
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'mean_force': np.mean(forces),
        'std_force': np.std(forces),
        'mean_velocity': np.mean(velocities),
        'std_velocity': np.std(velocities),
        'success_rate': np.mean(successes),
        'errors': errors,
    }


def evaluate_generalization(model, test_stiffnesses, target_distance, num_episodes):
    print(f"\nEvaluating on {len(test_stiffnesses)} test stiffnesses...")

    results = {}
    for stiffness in test_stiffnesses:
        print(f"  Testing {stiffness:.0f} N/m...", end=' ')
        results[stiffness] = evaluate_on_stiffness(
            model, stiffness, target_distance, num_episodes
        )
        print(f"Error: {results[stiffness]['mean_error']:.2f}cm, "
              f"Success: {results[stiffness]['success_rate'] * 100:.0f}%")

    return results


def plot_training_curves(all_training_metrics, config):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Training (1cm tolerance)', fontsize=16, fontweight='bold')

    metrics_to_plot = [
        ('errors', 'Distance Error (cm)', axes[0, 0]),
        ('rewards', 'Episode Reward', axes[0, 1]),
        ('peak_forces', 'Peak Force (N)', axes[1, 0]),
        ('peak_velocities', 'Peak Velocity (m/s)', axes[1, 1]),
    ]

    for protocol_name, metrics in all_training_metrics.items():
        for metric_key, ylabel, ax in metrics_to_plot:
            window = 20
            data = metrics[metric_key]
            episodes = metrics['episodes']

            if len(data) > window:
                smoothed = pd.Series(data).rolling(window=window, min_periods=1).mean()
                ax.plot(episodes, smoothed, label=protocol_name, linewidth=2, alpha=0.8)

            ax.set_xlabel('Episode')
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(config['plots_dir'], 'challenging_training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to {plot_path}")
    plt.close()


def plot_generalization_heatmap(all_generalization_results, test_stiffnesses, config):
    protocols = list(all_generalization_results.keys())
    error_matrix = np.zeros((len(protocols), len(test_stiffnesses)))

    for i, protocol in enumerate(protocols):
        for j, stiffness in enumerate(test_stiffnesses):
            error_matrix[i, j] = all_generalization_results[protocol][stiffness]['mean_error']

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(
        error_matrix,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn_r',
        xticklabels=[f'{s:.0f}' for s in test_stiffnesses],
        yticklabels=protocols,
        cbar_kws={'label': 'Mean Error (cm)'},
        ax=ax,
        vmin=0,
        vmax=10
    )
    ax.set_xlabel('Test Stiffness (N/m)', fontsize=12)
    ax.set_ylabel('Training Protocol', fontsize=12)
    ax.set_title('CHALLENGING Generalization (1cm tolerance, 50-600 N/m medical range)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(config['plots_dir'], 'challenging_generalization_heatmap.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved generalization heatmap to {plot_path}")
    plt.close()


def plot_success_rates(all_generalization_results, test_stiffnesses, config):
    fig, ax = plt.subplots(figsize=(14, 7))

    for protocol_name, results in all_generalization_results.items():
        success_rates = [results[s]['success_rate'] * 100 for s in test_stiffnesses]
        ax.plot(test_stiffnesses, success_rates, marker='o', linewidth=2,
                label=protocol_name, markersize=6, alpha=0.8)

    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    ax.set_xlabel('Test Stiffness (N/m)', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rates (1cm tolerance)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])

    plt.tight_layout()
    plot_path = os.path.join(config['plots_dir'], 'challenging_success_rates.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved success rates to {plot_path}")
    plt.close()


def plot_protocol_comparison(all_generalization_results, config):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Protocol Comparison on Unseen Stiffnesses (1cm tolerance)',
                 fontsize=16, fontweight='bold')

    protocols = list(all_generalization_results.keys())

    metrics = {
        'Error (cm)': [],
        'Peak Force (N)': [],
        'Success Rate (%)': []
    }

    for protocol in protocols:
        errors = []
        forces = []
        success_rates = []

        for stiffness_results in all_generalization_results[protocol].values():
            errors.extend(stiffness_results['errors'])
            forces.append(stiffness_results['mean_force'])
            success_rates.append(stiffness_results['success_rate'] * 100)

        metrics['Error (cm)'].append(errors)
        metrics['Peak Force (N)'].append(forces)
        metrics['Success Rate (%)'].append(success_rates)

    for ax, (metric_name, data) in zip(axes, metrics.items()):
        bp = ax.boxplot(data, labels=protocols, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_xticklabels(protocols, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = os.path.join(config['plots_dir'], 'challenging_protocol_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved protocol comparison to {plot_path}")
    plt.close()


def run_complete_pipeline():
    print("\n" + "=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training: 150k steps")
    print(f"Tolerance: 1cm")
    print(f"Range: 50-600 N/m")
    print(f"Protocols: 5")
    print(f"Test points: 14 untrained stiffnesses")
    print("=" * 80)

    os.makedirs(CONFIG['results_dir'], exist_ok=True)
    os.makedirs(CONFIG['models_dir'], exist_ok=True)
    os.makedirs(CONFIG['plots_dir'], exist_ok=True)

    all_training_metrics = {}
    all_generalization_results = {}
    all_trained_stiffnesses = {}

    print("\n" + "=" * 80)
    print("PHASE 1: TRAINING 5 PROTOCOLS")
    print("=" * 80)

    for protocol_name, protocol_config in CONFIG['protocols'].items():
        model, metrics, trained_stiffnesses = train_protocol(
            protocol_name, protocol_config, CONFIG
        )
        all_training_metrics[protocol_name] = metrics
        all_trained_stiffnesses[protocol_name] = trained_stiffnesses

    print("\n" + "=" * 80)
    print("PHASE 2: EVALUATING GENERALIZATION")
    print("=" * 80)

    for protocol_name in CONFIG['protocols'].keys():
        print(f"\n{protocol_name}:")
        model_path = os.path.join(CONFIG['models_dir'], f"{protocol_name}.zip")
        model = SAC.load(model_path)

        results = evaluate_generalization(
            model,
            CONFIG['test_stiffnesses'],
            CONFIG['target_distance'],
            CONFIG['eval_episodes_per_stiffness']
        )
        all_generalization_results[protocol_name] = results

    print("\n" + "=" * 80)
    print("PHASE 3: GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_training_curves(all_training_metrics, CONFIG)
    plot_generalization_heatmap(all_generalization_results, CONFIG['test_stiffnesses'], CONFIG)
    plot_success_rates(all_generalization_results, CONFIG['test_stiffnesses'], CONFIG)
    plot_protocol_comparison(all_generalization_results, CONFIG)

    print("\n" + "=" * 80)
    print("PHASE 4: SAVING RESULTS")
    print("=" * 80)

    results_summary = {
        'config': CONFIG,
        'challenging_settings': {
            'tolerance': '1cm (stricter)',
            'range': '100-1500 N/m (wider)',
            'protocols': 8,
            'training_steps': 250000
        },
        'trained_stiffnesses': all_trained_stiffnesses,
        'generalization_results': {
            protocol: {
                str(stiff): {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                             for k, v in results.items() if k != 'errors'}
                for stiff, results in protocol_results.items()
            }
            for protocol, protocol_results in all_generalization_results.items()
        },
        'timestamp': datetime.now().isoformat()
    }

    results_path = os.path.join(CONFIG['results_dir'], 'challenging_results_summary.json')
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"Saved results summary to {results_path}")

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print("\nBest Performing Protocol (lowest error on unseen stiffnesses):")
    mean_errors = {}
    mean_success = {}
    for protocol, results in all_generalization_results.items():
        all_errors = [r['mean_error'] for r in results.values()]
        all_success = [r['success_rate'] for r in results.values()]
        mean_errors[protocol] = np.mean(all_errors)
        mean_success[protocol] = np.mean(all_success)

    best_protocol = min(mean_errors, key=mean_errors.get)
    print(f"{best_protocol}")
    print(f"Error: {mean_errors[best_protocol]:.2f} cm")
    print(f"Success Rate: {mean_success[best_protocol] * 100:.1f}%")

    print("\nAll protocols ranked by generalization error:")
    for i, (protocol, error) in enumerate(sorted(mean_errors.items(), key=lambda x: x[1]), 1):
        print(f"  {i}. {protocol}: {error:.2f}cm error, {mean_success[protocol] * 100:.1f}% success")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to:")
    print(f"  - Models: {CONFIG['models_dir']}/")
    print(f"  - Plots: {CONFIG['plots_dir']}/")
    print(f"  - Data: {CONFIG['results_dir']}/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_complete_pipeline()

