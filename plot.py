import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def plot_results():
    plt.figure(figsize=(10, 6))
    for log_file in glob.glob("logs/*.csv"):
        label = os.path.basename(log_file).replace(".monitor.csv", "")
        data = pd.read_csv(log_file, skiprows=1)
        plt.plot(data.index, data['r'].rolling(window=50).mean(), label=label)
    
    plt.title("Learning Curves: Baseline vs Shaped Reward")
    plt.xlabel("Episodes")
    plt.ylabel("Mean Reward")
    plt.legend()
    plt.savefig("reward_comparison.png")
    plt.show()

if __name__ == "__main__":
    plot_results()