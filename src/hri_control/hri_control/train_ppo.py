import rclpy
from stable_baselines3 import PPO 
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from hri_control.hri_env_final import HriEnv
import os

#creating environments and ppo paths
def main():
    rclpy.init()
    env = HriEnv()
    #for tensoboard to show reward graphs
    env = Monitor(env)

    save_path = "./checkpoints_ppo/"
    log_path = "./eval_logs_ppo/"
    tb_log = "./ppo_hri_tensorboard/"
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

#backup after 50k steps to resume if crash
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_path,
        name_prefix="ppo_hri" 
    )

    eval_env = HriEnv()
    eval_env = Monitor(eval_env)
    #to handle overfitting, after every 25k steps test on test env
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model_ppo/",
        log_path=log_path,
        eval_freq=25000,
        deterministic=True,
        render=False
    )
    callback = CallbackList([checkpoint_callback, eval_callback])

   #Hyperparameters for PPO
    print("Initializing PPO model")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,     
        n_steps=2048,      #collecteing data for 2048 steps and then update    
        batch_size=64,         
        n_epochs=10,            
        gamma=0.99,
        gae_lambda=0.95,  #for bias variance tradeoff      
        clip_range=0.2,      #only allow 20% change from old polcy  
        ent_coef=0.0,        #no bonus
        tensorboard_log=tb_log
    )

    print("Starting PPO training")
    
    model.learn(
        total_timesteps=700000,  
        callback=callback
    )

    model.save("ppo_hri_final")
    
    print("PPO Training Finished!")
    rclpy.shutdown()

if __name__ == "__main__":
    main()
