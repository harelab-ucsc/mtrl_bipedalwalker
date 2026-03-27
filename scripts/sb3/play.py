import gymnasium as gym
from stable_baselines3 import PPO

from lib.utils.paths import MODELS_DIR
from lib.wrappers.bipedal_walker.test import Test_Wrapper

# =========================================

EXPERIMENT_NAME = "3_25_2026/basic_walker_5-00_39_40-2026_03_26"
MODEL_CHECKPOINT = "best/best_model"

# =========================================

env = gym.make("BipedalWalker-v3", render_mode="human")
wrap_env = Test_Wrapper(env, plotting=True)
wrap_env.reset()

model_path = MODELS_DIR / f"{EXPERIMENT_NAME}/{MODEL_CHECKPOINT}.zip"
model = PPO.load(model_path, env=wrap_env)

episodes = 5

for ep in range(episodes):
    obs, _ = wrap_env.reset()
    done = False
    total_rewards = 0
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info, _ = wrap_env.step(action)
        total_rewards += float(rewards)
        wrap_env.render()
    print(total_rewards)
