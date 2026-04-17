from stable_baselines3 import PPO
from astro_env import AstroBalanceEnv

# 1. Create the environment
env = AstroBalanceEnv()

# 2. Create the Brain (PPO Algorithm)
# We use 'MlpPolicy' which is a standard Neural Network
model = PPO("MlpPolicy", env, verbose=1)

# 3. Train the AI for 50,000 steps
print("AI is starting to learn how to balance the satellite...")
model.learn(total_timesteps=50000)

# 4. Save the Brain
model.save("astro_balance_brain")
print("Training complete! Brain saved as astro_balance_brain.zip")