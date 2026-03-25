import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pymunk
import pymunk.pygame_util
import pygame

class DoublePendulumEnv(gym.Env):
    def __init__(self, reward_type='shaped', render_mode=None):
        super(DoublePendulumEnv, self).__init__()
        self.reward_type = reward_type
        self.render_mode = render_mode
        
        # [cart_x, cart_vx, pole1_theta, pole1_omega, pole2_theta, pole2_omega]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        self.screen = None
        self.clock = None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.space = pymunk.Space()
        self.space.gravity = (0, -981)
        
        # Physics setup: Cart and two poles
        mass = 1.0
        moment = pymunk.moment_for_poly(mass, [(-50, -10), (50, -10), (50, 10), (-50, 10)])
        self.cart = pymunk.Body(mass, moment)
        self.cart.position = (400, 200)
        
        # Joint and pole definitions would go here...
        # Returning dummy initial state for structure
        return np.zeros(6, dtype=np.float32), {}

    def step(self, action):
        force = action[0] * 1000
        self.cart.apply_force_at_local_point((force, 0))
        self.space.step(1/60.0)
        
        obs = self._get_obs()
        reward = self._calculate_reward(obs, action)
        terminated = bool(abs(obs[2]) > 0.8 or abs(obs[4]) > 0.8)
        
        return obs, reward, terminated, False, {}

    def _get_obs(self):
        # Implementation to extract state from pymunk bodies
        return np.zeros(6, dtype=np.float32)

    def _calculate_reward(self, obs, action):
        theta1, theta2 = obs[2], obs[4]
        baseline = np.cos(theta1) + np.cos(theta2)
        
        if self.reward_type == 'baseline':
            return baseline
        
        # Shaped reward components
        cart_penalty = -0.1 * abs(obs[0] - 400) / 400
        velocity_penalty = -0.01 * (abs(obs[3]) + abs(obs[5]))
        action_penalty = -0.001 * (action[0]**2)
        
        return baseline + cart_penalty + velocity_penalty + action_penalty