# train_ppo.py

import rclpy
from stable_baselines3 import PPO  # <--- USING PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from hri_control.hri_env_final import HriEnv
import os

def main():
    # Initialize ROS2
    rclpy.init()

    # Create environment
    env = HriEnv()
    env = Monitor(env)

    # -----------------------------
    # PPO-SPECIFIC PATHS (Separated from SAC)
    # -----------------------------
    save_path = "./checkpoints_ppo/"
    log_path = "./eval_logs_ppo/"
    tb_log = "./ppo_hri_tensorboard/"
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # -----------------------------
    # CALLBACKS
    # -----------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_path,
        name_prefix="ppo_hri" 
    )

    eval_env = HriEnv()
    eval_env = Monitor(eval_env)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model_ppo/",
        log_path=log_path,
        eval_freq=25000,
        deterministic=True,
        render=False
    )
    callback = CallbackList([checkpoint_callback, eval_callback])

    # -----------------------------
    # PPO HYPERPARAMETERS
    # -----------------------------
    print("Initializing PPO model...")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,     # Standard stable rate
        n_steps=2048,           # Steps to collect before updating (On-Policy buffer)
        batch_size=64,          # PPO usually prefers smaller batches than SAC
        n_epochs=10,            # Reuse data 10 times for optimization
        gamma=0.99,
        gae_lambda=0.95,        # Generalized Advantage Estimation
        clip_range=0.2,         # Clip updates to prevent drastic policy changes
        ent_coef=0.0,           # PPO handles exploration differently (usually 0.0 or small)
        tensorboard_log=tb_log
    )

    # -----------------------------
    # TRAINING
    # -----------------------------

    print("Starting PPO training for 500,000 steps...")
    
    model.learn(
        total_timesteps=500000,   # Match SAC exactly for fair comparison
        callback=callback
    )

    model.save("ppo_hri_final")
    
    print("PPO Training Finished!")
    rclpy.shutdown()

if __name__ == "__main__":
    main()
