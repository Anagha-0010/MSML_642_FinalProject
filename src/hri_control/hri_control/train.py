# This file is train.py
# This is the main script you will run to start the RL training.

import rclpy
from hri_control.hri_env_final import HriEnv  # Import our custom environment
from stable_baselines3 import SAC
# --- 1. ADD THIS IMPORT ---
from stable_baselines3.common.callbacks import CheckpointCallback
import time

def main(args=None):
    # Initialize ROS2
    rclpy.init(args=args)

    print("--- HRI Training Script Started ---")

    # --- 1. Create the Environment ---
    print("Creating HRI Environment...")
    env = HriEnv()
    print("Environment created.")

    # --- 2. CREATE THE CHECKPOINT CALLBACK ---
    # This will save a checkpoint of your model every 10,000 steps
    # in a folder named 'sac_hri_checkpoints/'
    checkpoint_callback = CheckpointCallback(
      save_freq=10000,
      save_path="./sac_hri_checkpoints/",
      name_prefix="rl_model"
    )

    # --- 3. Initialize the SAC Agent ---
    print("Initializing SAC model...")
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,  # This will print training progress
        tensorboard_log="./sac_hri_tensorboard/"
    )
    print("SAC model initialized.")

    # --- 4. Train the Agent ---
    total_timesteps = 2000
    print(f"Starting training for {total_timesteps} timesteps...")
    
    start_time = time.time()
    
    # --- 5. ADD THE CALLBACK TO THE .learn() METHOD ---
    # The agent will now save its progress automatically
    model.learn(
        total_timesteps=total_timesteps, 
        log_interval=100, 
        callback=checkpoint_callback # <-- THIS IS THE NEW ARGUMENT
    )
    
    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")

    # --- 5. Save the Trained Model ---
    model_save_path = "./sac_hri_model"
    print(f"Saving FINAL trained model to {model_save_path}")
    model.save(model_save_path)

    # --- 6. Clean Up ---
    print("Shutting down...")
    env.close()
    rclpy.shutdown()
    print("--- HRI Training Script Finished ---")


if __name__ == '__main__':
    main()
