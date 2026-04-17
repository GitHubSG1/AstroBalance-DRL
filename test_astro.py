import gymnasium as gym
from stable_baselines3 import PPO
from astro_env import AstroBalanceEnv
import matplotlib.pyplot as plt

# 1. Load the environment and the trained "Brain"
env = AstroBalanceEnv()
model = PPO.load("astro_balance_brain")

# 2. Run a test simulation
obs, _ = env.reset()
history = []

print("Testing the trained AI...")
for _ in range(200):
    # Ask the AI for the best torque
    action, _states = model.predict(obs, deterministic=True)
    
    # Apply it to the satellite
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Record the angle to plot it
    history.append(obs[0]) 
    
    if terminated:
        print("Satellite crashed during test!")
        break

# 3. Plot the results (The Engineering Proof)
plt.plot(history)
plt.axhline(y=0, color='r', linestyle='--') # The target (0 degrees)
plt.title("Satellite Orientation (RL Control)")
plt.xlabel("Time Steps (50ms)")
plt.ylabel("Angle (Radians)")
plt.grid(True)
plt.show()