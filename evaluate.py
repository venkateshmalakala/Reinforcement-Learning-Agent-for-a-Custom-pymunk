import argparse
import os
import pygame
import imageio
import numpy as np
from stable_baselines3 import PPO
from environment import DoublePendulumEnv

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO Agent")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the saved model (.zip)")
    parser.add_argument('--gif_path', type=str, default=None, help="Path to save evaluation GIF (e.g., media/agent_final.gif)")
    args = parser.parse_args()

    env = DoublePendulumEnv()
    
    # Load the trained model
    model = PPO.load(args.model_path, env=env)
    obs = env.reset()
    
    frames = []
    print("Starting evaluation...")
    
    for step in range(1000):
        # Predict action (deterministic=True is best for evaluation)
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # Render the environment
        env.render()
        
        # Capture frames for GIF if requested
        if args.gif_path:
            surface = pygame.display.get_surface()
            # Convert Pygame surface to imageio compatible array
            frame = pygame.surfarray.array3d(surface)
            frame = np.rot90(frame, k=-1)
            frame = np.flipud(frame)
            frames.append(frame)

        if done:
            obs = env.reset()

    env.close()

    # Save the GIF
    if args.gif_path and frames:
        os.makedirs(os.path.dirname(args.gif_path), exist_ok=True)
        imageio.mimsave(args.gif_path, frames, fps=60)
        print(f"Successfully saved evaluation GIF to {args.gif_path}")

if __name__ == '__main__':
    main()
    