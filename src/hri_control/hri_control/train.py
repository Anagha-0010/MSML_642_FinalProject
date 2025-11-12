#!/usr/bin/env python3

import rclpy
import os
from hri_control.hri_env import HRI_Env
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

def main(args=None):
    rclpy.init(args=args)
    
    # --- 1. Create the Gym Environment ---
    # We must wrap it in a DummyVecEnv for SB3
    env = DummyVecEnv([lambda: HRI_Env()])
    
    # --- 2. Create the SAC Agent ---
    # We'll log to a 'sac_tensorboard/' directory
    tensorboard_log = os.path.join(os.path.expanduser('~'), 'hri_project_ws', 'sac_tensorboard')
    os.makedirs(tensorboard_log, exist_ok=True)
    
    # 'MlpPolicy' is the default neural network (a Multi-Layer Perceptron)
    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=tensorboard_log,
        learning_rate=0.0003,
        buffer_size=100000,
        batch_size=256,
        gamma=0.99
    )
    
    # --- 3. Train the Agent ---
    # 50,000 steps is enough for this simple test.
    # log_interval=10 means it will print stats every 10 episodes
    print("--- Starting Phase 2 Training (Learn to do nothing) ---")
    model.learn(total_timesteps=50000, log_interval=10)
    
    # --- 4. Save the Model ---
    model_save_path = os.path.join(os.path.expanduser('~'), 'hri_project_ws', 'sac_phase2_model')
    model.save(model_save_path)
    
    print(f"--- Training Complete. Model saved to {model_save_path} ---")

    # Clean up
    env.close()
    rclpy.shutdown()

if _name_ == '_main_':
    main()
