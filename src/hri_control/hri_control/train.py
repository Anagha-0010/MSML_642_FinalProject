# train.py

import rclpy
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from hri_control.hri_env_final import HriEnv

def main():
    rclpy.init()

    env = HriEnv()

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./checkpoints/",
        name_prefix="sac_hri"
    )

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./sac_hri_tensorboard/"
    )

    model.learn(
        total_timesteps=300000,
        callback=checkpoint_callback
    )

    model.save("sac_hri_final")

    rclpy.shutdown()


if __name__ == "__main__":
    main()

