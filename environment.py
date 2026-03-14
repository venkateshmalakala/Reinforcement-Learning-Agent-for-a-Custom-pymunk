import gym
from gym import spaces
import numpy as np
import pymunk
import pymunk.pygame_util
import pygame

class DoublePendulumEnv(gym.Env):
    """Custom Environment that follows gym interface for a Double Pendulum"""
    metadata = {'render.modes': ['human']}

    def __init__(self, reward_type='shaped'):
        super(DoublePendulumEnv, self).__init__()
        self.reward_type = reward_type
        
        # Action space: Continuous force applied to the cart [-1.0, 1.0]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation space: 6 variables
        # [cart_x, cart_vx, pole1_theta, pole1_omega, pole2_theta, pole2_omega]
        high = np.array([np.inf] * 6, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        
        # Pygame & Pymunk setup
        self.screen = None
        self.clock = None
        self.space = None
        self.draw_options = None
        
        self.reset()

    def reset(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, -980) # Gravity points downwards
        
        # Setup Physics Bodies (Cart, Pole 1, Pole 2)
        # 1. Cart
        mass_cart = 1.0
        size_cart = (50, 30)
        moment_cart = pymunk.moment_for_box(mass_cart, size_cart)
        self.cart_body = pymunk.Body(mass_cart, moment_cart)
        self.cart_body.position = (300, 300)
        self.cart_shape = pymunk.Poly.create_box(self.cart_body, size_cart)
        self.cart_shape.color = (255, 0, 0, 255)
        
        # 2. Track (Static)
        static_body = self.space.static_body
        self.track_joint = pymunk.GrooveJoint(static_body, self.cart_body, (50, 300), (550, 300), (0, 0))
        
        # 3. Pole 1
        mass_pole1 = 0.5
        size_pole1 = (10, 100)
        moment_pole1 = pymunk.moment_for_box(mass_pole1, size_pole1)
        self.pole1_body = pymunk.Body(mass_pole1, moment_pole1)
        self.pole1_body.position = (300, 350)
        self.pole1_shape = pymunk.Poly.create_box(self.pole1_body, size_pole1)
        
        # Joint Cart -> Pole 1
        self.joint1 = pymunk.PivotJoint(self.cart_body, self.pole1_body, (300, 300))
        
        # 4. Pole 2
        mass_pole2 = 0.5
        size_pole2 = (10, 100)
        moment_pole2 = pymunk.moment_for_box(mass_pole2, size_pole2)
        self.pole2_body = pymunk.Body(mass_pole2, moment_pole2)
        self.pole2_body.position = (300, 450)
        self.pole2_shape = pymunk.Poly.create_box(self.pole2_body, size_pole2)
        
        # Joint Pole 1 -> Pole 2
        self.joint2 = pymunk.PivotJoint(self.pole1_body, self.pole2_body, (300, 400))
        
        # Add to space
        self.space.add(self.cart_body, self.cart_shape, self.track_joint)
        self.space.add(self.pole1_body, self.pole1_shape, self.joint1)
        self.space.add(self.pole2_body, self.pole2_shape, self.joint2)
        
        return self._get_obs()

    def _get_obs(self):
        # Normalize cart position relative to center (300)
        cart_x = (self.cart_body.position.x - 300) / 100.0 
        cart_vx = self.cart_body.velocity.x / 100.0
        
        pole1_theta = self.pole1_body.angle
        pole1_omega = self.pole1_body.angular_velocity
        
        pole2_theta = self.pole2_body.angle
        pole2_omega = self.pole2_body.angular_velocity
        
        return np.array([cart_x, cart_vx, pole1_theta, pole1_omega, pole2_theta, pole2_omega], dtype=np.float32)

    def step(self, action):
        # Apply force based on action
        force_magnitude = float(action[0]) * 1000.0 
        self.cart_body.apply_force_at_local_point((force_magnitude, 0), (0, 0))
        
        # Step physics
        dt = 1.0 / 60.0
        self.space.step(dt)
        
        obs = self._get_obs()
        cart_x, cart_vx, theta1, omega1, theta2, omega2 = obs
        
        # Calculate Reward
        baseline_reward = np.cos(theta1) + np.cos(theta2)
        
        if self.reward_type == 'shaped':
            center_penalty = abs(cart_x) * 0.1
            velocity_penalty = (abs(omega1) + abs(omega2)) * 0.01
            action_penalty = (action[0]**2) * 0.001
            reward = baseline_reward - center_penalty - velocity_penalty - action_penalty
        else:
            reward = baseline_reward
            
        # Done condition (if it falls too far or goes off screen)
        done = bool(
            abs(cart_x) > 2.5 or 
            abs(theta1) > np.pi/2 or 
            abs(theta2) > np.pi/2
        )
        
        return obs, float(reward), done, {}

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((600, 600))
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            
        self.screen.fill((255, 255, 255))
        self.space.debug_draw(self.draw_options)
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None