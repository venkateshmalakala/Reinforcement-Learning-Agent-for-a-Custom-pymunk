import os
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy

def generate_plot():
    plt.figure(figsize=(10, 6))
    
    def plot_log_data(log_dir, label):
        if os.path.exists(log_dir):
            df = load_results(log_dir)
            if len(df) > 0:
                x, y = ts2xy(df, 'timesteps')
                # Apply a rolling window to smooth the noisy RL data
                y_smoothed = pd.Series(y).rolling(window=50, min_periods=1).mean()
                plt.plot(x, y_smoothed, label=label, linewidth=2)

    # Plot both reward types
    plot_log_data('logs/baseline', 'Baseline Reward')
    plot_log_data('logs/shaped', 'Shaped Reward')

    # Formatting the chart
    plt.title('Learning Curve Comparison: Baseline vs. Shaped Reward')
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward')
    plt.legend()
    plt.grid(True)
    
    # Save to root directory
    plt.savefig('reward_comparison.png')
    print("Plot successfully generated and saved as 'reward_comparison.png'")

if __name__ == '__main__':
    generate_plot()