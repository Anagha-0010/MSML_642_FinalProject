# test.py

import rclpy
from hri_control.hri_env_final import HriEnv
from stable_baselines3 import SAC
import time

MODEL_PATH = "/home/anagha/MSML_642_FinalProject/checkpoints/sac_hri_250000_steps.zip"

def main():
    rclpy.init()

    print("\n==============================")
    print("   HRI TESTING AGENT STARTED")
    print("==============================\n")

    env = HriEnv()

    print(f"Loading model from: {MODEL_PATH}")
    model = SAC.load(MODEL_PATH)     # <-- IMPORTANT: DO NOT pass env here
    print("Model loaded.")

    obs, _ = env.reset()
    step = 0

    while rclpy.ok():

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        print(f"[Step {step}] Dist = {info.get('dist')} | Reward = {reward:.3f}")
        step += 1

        if terminated or truncated:
            print("\n--- Episode finished. Resetting... ---\n")
            obs, _ = env.reset()

    env.close()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

