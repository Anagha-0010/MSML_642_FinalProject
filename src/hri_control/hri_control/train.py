# This file is train.py
# This is the main script you will run to start the RL training.

import rclpy
from hri_control.hri_env import HriEnv  # Import our custom environment
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
import time

def main(args=None):
    # Initialize ROS2
    rclpy.init(args=args)

    print("--- HRI Training Script Started ---")

    # --- 1. Create the Environment ---
    # We must create the env and spin it in a background thread
    # so that ROS2 callbacks (like /joint_states) can be processed.
    
    print("Creating HRI Environment...")
    env = HriEnv()
    
    # We use a MultiThreadedExecutor to handle the env's node
    # and the SAC agent's operations concurrently.
    # Note: We don't spin() here, we let the agent's .learn() do the work.
    # We pass the node to the agent, which is a bit unusual,
    # but for sb3, we just need the env to be able to spin itself.
    # The `step` and `reset` functions in HriEnv handle the spinning.
    
    print("Environment created.")

    # --- 2. Check the Environment (Optional but Recommended) ---
    # This test makes sure your custom env follows the Gymnasium API
    # try:
    #     print("Checking environment...")
    #     check_env(env)
    #     print("Environment check passed!")
    # except Exception as e:
    #     print(f"Environment check failed! {e}")
    #     env.close()
    #     rclpy.shutdown()
    #     return

    # --- 3. Initialize the SAC Agent ---
    # We are using the "MlpPolicy" (a standard neural network)
    # and the SAC algorithm.
    print("Initializing SAC model...")
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,  # This will print training progress
        tensorboard_log="./sac_hri_tensorboard/"
    )
    print("SAC model initialized.")

    # --- 4. Train the Agent ---
    # This is the main training loop.
    # It will run for 100,000 steps, which is a good start.
    total_timesteps = 100000
    print(f"Starting training for {total_timesteps} timesteps...")
    
    start_time = time.time()
    
    model.learn(total_timesteps=total_timesteps, log_interval=100)
    
    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")

    # --- 5. Save the Trained Model ---
    model_save_path = "./sac_hri_model"
    print(f"Saving trained model to {model_save_path}")
    model.save(model_save_path)

    # --- 6. Clean Up ---
    print("Shutting down...")
    env.close()
    rclpy.shutdown()
    print("--- HRI Training Script Finished ---")


if __name__ == '__main__':
    main()
