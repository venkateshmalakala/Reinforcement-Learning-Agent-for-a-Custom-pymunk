import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from environment import DoublePendulumEnv

def main():
    parser = argparse.ArgumentParser(description="Train PPO Agent on Double Pendulum")
    parser.add_argument('--timesteps', type=int, default=200000, help="Total timesteps to train")
    parser.add_argument('--reward_type', type=str, choices=['baseline', 'shaped'], default='shaped', help="Reward function to use")
    parser.add_argument('--save_path', type=str, default='models/ppo_model', help="Path to save the trained model")
    args = parser.parse_args()

    # Create log directory for the specific reward type
    log_dir = f"logs/{args.reward_type}"
    os.makedirs(log_dir, exist_ok=True)

    # Initialize environment and wrap with Monitor for logging metrics
    env = DoublePendulumEnv(reward_type=args.reward_type)
    env = Monitor(env, log_dir)

    print(f"Starting training with '{args.reward_type}' reward for {args.timesteps} timesteps...")
    
    # Initialize PPO Agent using Multi-Layer Perceptron (MlpPolicy)
    model = PPO('MlpPolicy', env, verbose=1)
    
    # Train
    model.learn(total_timesteps=args.timesteps)
    
    # Save the final model
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    model.save(args.save_path)
    print(f"Training complete. Model saved to {args.save_path}.zip")

if __name__ == '__main__':
    main()