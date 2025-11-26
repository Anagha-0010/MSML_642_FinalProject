# train.py

import rclpy
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from hri_control.hri_env_final import HriEnv


def main():
    # Initialize ROS2
    rclpy.init()

    # Create environment
    env = HriEnv()
    env = Monitor(env)   # adds episode logging

    # -----------------------------
    # CALLBACKS
    # -----------------------------

    # Save periodic checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./checkpoints/",
        name_prefix="sac_hri_fresh_again"  # Changed name to avoid overwriting old files
    )

    # Evaluation env (no exploration noise)
    eval_env = HriEnv()
    eval_env = Monitor(eval_env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./eval_logs/",
        eval_freq=25000,
        deterministic=True,
        render=False
    )

    callback = CallbackList([checkpoint_callback, eval_callback])

    # -----------------------------
    # SAC Hyperparameters (FRESH START)
    # -----------------------------

    print("Initializing NEW model (Fresh Start)...")
    model_path = "./checkpoints/sac_hri_fresh_200000_steps.zip"
    model = SAC.load(
        model_path,
        env=env,
        print_system_info=True,
        #verbose=1,
        #buffer_size=300000,      
        #batch_size=256,         
        #learning_rate=3e-4,      
        #gamma=0.99,              
        #tau=0.005,               
        #train_freq=1,            
        #gradient_steps=1,        
        #ent_coef="auto",         
        tensorboard_log="./sac_hri_tensorboard/"
    )

    # -----------------------------
    # TRAINING
    # -----------------------------

    print("Starting training for 500,000 steps...")
    
    model.learn(
        total_timesteps=500000,   # Sufficient time to learn the new reward
        callback=callback,
        reset_num_timesteps=False
    )

    model.save("sac_hri_final_fresh")

    # Shutdown ROS2
    rclpy.shutdown()


if __name__ == "__main__":
    main()
