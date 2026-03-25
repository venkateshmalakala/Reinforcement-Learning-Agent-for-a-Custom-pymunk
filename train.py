import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from environment import DoublePendulumEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reward_type', type=str, default='shaped')
    parser.add_argument('--timesteps', type=int, default=200000)
    parser.add_argument('--save_path', type=str, default='models/ppo_model')
    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    env = DoublePendulumEnv(reward_type=args.reward_type)
    env = Monitor(env, f"logs/{args.reward_type}")

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save_path)

if __name__ == "__main__":
    main()