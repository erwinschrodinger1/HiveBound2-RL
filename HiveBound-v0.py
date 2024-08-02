import gymnasium as gym
from stable_baselines3 import PPO
import time
import os

from rl_environment import HiveBoundEnv
import pygame

models_dir = f"models/model-{int(time.time())}"
log_dir = f"logs/log-{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


env = HiveBoundEnv()
env.reset()


model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
TIMESTEPS = 10000

for i in range(1, 10000):
    model.learn(
        total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=log_dir
    )
    model.save(f"{models_dir}/{TIMESTEPS*i}")

env.close()
