import gymnasium as gym
from gymnasium import spaces
import numpy as np

class AstroBalanceEnv(gym.Env):
    def __init__(self):
        super(AstroBalanceEnv, self).__init__()
        
        # Action: Motor Torque (-1.0 to 1.0)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation: [Angle, Angular_Velocity]
        # Angle is between -pi and pi. Velocity is limited to -10 to 10.
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -10.0]), 
            high=np.array([np.pi, 10.0]), 
            dtype=np.float32
        )
        
        self.state = None
        self.dt = 0.05  # Time step: 50ms (same as a real-time controller)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start at a random wobbly angle
        self.state = np.array([np.random.uniform(-0.5, 0.5), 0.0], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        angle, velocity = self.state
        torque = action[0]
        
        # 1. ADD DISTURBANCE (Solar Wind / Grid Noise)
        # This shows your AI is "Robust"
        disturbance = np.random.uniform(-0.02, 0.02)
        
        # 2. EE MATH
        inertia = 1.0 
        acceleration = (torque + disturbance) / inertia
        
        new_velocity = velocity + acceleration * self.dt
        new_angle = angle + new_velocity * self.dt
        
        self.state = np.array([new_angle, new_velocity], dtype=np.float32)

        # 3. SMART INVERTER OPTIMIZATION (Efficiency Reward)
        # Penalty for high torque = SiC Switching Loss optimization
        energy_penalty = 0.1 * (torque**2) 
        reward = (1.0 - abs(new_angle)) - energy_penalty
        
        terminated = bool(abs(new_angle) > np.pi/2)
        return self.state, reward, terminated, False, {}

print("Digital Twin 'Astro-Balance' is coded!")