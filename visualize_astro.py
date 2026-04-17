import gymnasium as gym
from stable_baselines3 import PPO
from astro_env import AstroBalanceEnv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# 1. Setup
env = AstroBalanceEnv()
model = PPO.load("astro_balance_brain")
obs, _ = env.reset()

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.grid(True)

# Draw the "Satellite" (a rectangle) and the "Target" (a line)
satellite_body, = ax.plot([], [], 'b-', lw=4, label="Satellite")
target_line = ax.axvline(0, color='r', linestyle='--', alpha=0.3, label="Target (0°)")
ax.legend()

def init():
    satellite_body.set_data([], [])
    return satellite_body,

def update(frame):
    global obs
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, _, _ = env.step(action)
    
    angle = obs[0]
    
    # Calculate the points of a line representing the satellite
    x = [0.8 * np.sin(angle), -0.8 * np.sin(angle)]
    y = [0.8 * np.cos(angle), -0.8 * np.cos(angle)]
    
    satellite_body.set_data(x, y)
    
    if terminated:
        obs, _ = env.reset()
        
    return satellite_body,

# This creates the dynamic animation
ani = animation.FuncAnimation(fig, update, frames=200, init_func=init, blit=True, interval=50)
plt.title("DRL Satellite Stabilization (Live Simulation)")
plt.show()