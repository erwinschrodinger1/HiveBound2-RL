import gymnasium as gym
from stable_baselines3 import PPO
import time
import os

from rl_environment import HiveBoundEnv
import pygame

models_dir = f"models"
model_path = f"{models_dir}/model-1722617210/250000"


env = HiveBoundEnv()
env.reset()


model = PPO.load(model_path, env=env)

episodes = 10

for ep in range(1, episodes):
    vec_env = model.get_env()
    obs = vec_env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info, _ = env.step(action)
        print(f"Reward: {reward}, info: {info}")

env.close()
