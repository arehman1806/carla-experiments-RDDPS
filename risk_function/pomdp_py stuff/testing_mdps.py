import gymnasium_envs

import gymnasium as gym

env = gym.make("perception_mdp/simple-v0")
obs, info = env.reset()

try:
    while True:
        if obs[0] > 40:
            detected = 0
        else:
            detected = 1
        obs, cost, terminated, _, info = env.step(detected)
        print(f"obs: {obs/100}, cost: {cost}, term: {terminated}")
        if terminated:
            obs, info = env.reset()
            print("-----------")
except KeyboardInterrupt:
    print("terminated")