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
        name_prefix="sac_hri"
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
    # SAC Hyperparameters
    # -----------------------------

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        buffer_size=300000,      # large replay memory
        batch_size=256,          # strong for control tasks
        learning_rate=3e-4,      # stable value for SAC
        gamma=0.99,              # discount factor
        tau=0.005,               # soft update
        train_freq=1,            # update every step
        gradient_steps=1,        # 1 update per environment step
        ent_coef="auto",         # automatic entropy tuning
        tensorboard_log="./sac_hri_tensorboard/"
    )

    # -----------------------------
    # TRAINING
    # -----------------------------

    model.learn(
        total_timesteps=400000,   # longer training for stability
        callback=callback
    )

    model.save("sac_hri_final")

    # Shutdown ROS2
    rclpy.shutdown()


if __name__ == "__main__":
    main()

