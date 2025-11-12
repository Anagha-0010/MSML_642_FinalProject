#!/usr/bin/env python3

import rclpy
import os
from hri_control.hri_env_final import HRI_Env_Final
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

def main(args=None):
    rclpy.init(args=args)
    
    # --- 1. Create the Gym Environment ---
    env = DummyVecEnv([lambda: HRI_Env_Final()])
    
    # --- 2. Create the SAC Agent ---
    tensorboard_log = os.path.join(os.path.expanduser('~'), 'hri_project_ws', 'sac_tensorboard_final')
    os.makedirs(tensorboard_log, exist_ok=True)
    
    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=tensorboard_log,
        learning_rate=0.0003,
        buffer_size=300000, # Larger buffer for a more complex task
        batch_size=256,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        learning_starts=10000 # Collect 10k random samples before training
    )
    
    # --- 3. Train the Agent ---
    # This task is MUCH harder. It needs 1,000,000+ steps.
    print("--- Starting Phase 3 Training (Handover Task) ---")
    model.learn(total_timesteps=1000000, log_interval=10)
    
    # --- 4. Save the Model ---
    model_save_path = os.path.join(os.path.expanduser('~'), 'hri_project_ws', 'sac_phase3_final_model')
    model.save(model_save_path)
    
    print(f"--- Training Complete. Model saved to {model_save_path} ---")

    # Clean up
    env.close()
    rclpy.shutdown()

if _name_ == '_main_':
    main()
