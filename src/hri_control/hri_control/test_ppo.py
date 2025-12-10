import rclpy
from hri_control.hri_env_final import HriEnv
from stable_baselines3 import PPO 
import time

MODEL_PATH = "ppo_hri_final.zip" 

def main():
    rclpy.init()
    print("TESTING PPO")
    env = HriEnv()

    print(f"Loading PPO model{MODEL_PATH}")
    try:
        model = PPO.load(MODEL_PATH)
    except FileNotFoundError:
        print("ERROR Model not found Check the filename.")
        return
    obs, _ = env.reset()
    step = 0

    while rclpy.ok():
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
