import gymnasium as gym
from stable_baselines3 import PPO

from lib.paths import MODELS_DIR

# =========================================

EXPERIMENT_NAME = "basic_walker_5-00_39_40-2026_03_26"
MODEL_CHECKPOINT = "best/best_model"

# =========================================

env = gym.make("BipedalWalker-v3", render_mode="human")
env.reset()

model_path = MODELS_DIR / f"{EXPERIMENT_NAME}/{MODEL_CHECKPOINT}.zip"
model = PPO.load(model_path, env=env)

episodes = 5

for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    total_rewards = 0
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info, _ = env.step(action)
        total_rewards += float(rewards)
        env.render()
    print(total_rewards)
