# test_ppo.py

import rclpy
from hri_control.hri_env_final import HriEnv
from stable_baselines3 import PPO  # <--- Critical change
import time

# Ensure this matches your saved PPO model name
MODEL_PATH = "ppo_hri_final.zip" 

def main():
    rclpy.init()
    print("\n=== TESTING PPO AGENT ===\n")
    
    env = HriEnv()

    print(f"Loading PPO model from: {MODEL_PATH}")
    try:
        model = PPO.load(MODEL_PATH)
    except FileNotFoundError:
        print("ERROR: Model not found! Check the filename.")
        return

    obs, _ = env.reset()
    step = 0

    while rclpy.ok():
        # Deterministic=True turns off training noise
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"[PPO Step {step}] Dist: {info.get('dist', 0.0):.3f} | Reward: {reward:.3f}")
        step += 1

        if terminated or truncated:
            print("--- Episode Finished ---")
            obs, _ = env.reset()
            step = 0
            time.sleep(1.0)

if __name__ == "__main__":
    main()
