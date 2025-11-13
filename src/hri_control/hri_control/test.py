# This file is test.py
# It loads a pre-trained model and runs it in inference mode.

import rclpy
from hri_control.hri_env import HriEnv  # Import our custom environment
from stable_baselines3 import SAC
import time

# --- DEFINE MODEL PATH ---
# Point this to your best checkpoint from the folder
MODEL_PATH = "./sac_hri_checkpoints/rl_model_100000_steps.zip"

def main(args=None):
    # Initialize ROS2
    rclpy.init(args=args)

    print("--- HRI Testing Script Started ---")

    # --- 1. Create the Environment ---
    print("Creating HRI Environment...")
    env = HriEnv()
    
    # --- 2. Load the Trained Model ---
    print(f"Loading trained model from {MODEL_PATH}...")
    model = SAC.load(MODEL_PATH, env=env)
    print("Model loaded.")

    # --- 3. Run the Model (No Training) ---
    obs, _ = env.reset()
    print("--- Running trained agent ---")
    
    # Loop forever, running the policy
    while rclpy.ok():
        # model.predict() gets the best action from the policy
        # deterministic=True means it won't explore, just pick the best move
        action, _states = model.predict(obs, deterministic=True)
        
        # Take the action in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print("Episode finished. Resetting...")
            obs, _ = env.reset()

    # --- 4. Clean Up ---
    print("Shutting down...")
    env.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
