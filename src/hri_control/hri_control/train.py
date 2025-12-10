import rclpy
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from hri_control.hri_env_final import HriEnv

#creating environment and add checkpoits to ensure if the system crashes we can train it again from the last checkpoint
def main():
    rclpy.init()
    env = HriEnv()
    env = Monitor(env)  
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./checkpoints/",
        name_prefix="sac_hri_final_final_again" 
    )

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

#SAC Hyperparameters
    print("Initializing model")
    model_path = "./checkpoints/sac_hri_final_200000_steps.zip"
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

    print("Starting training")
    
    model.learn(
        total_timesteps=700000,  
        callback=callback,
        reset_num_timesteps=False
    )

    model.save("sac_hri_final_final")

    # Shutdown ROS2
    rclpy.shutdown()


if __name__ == "__main__":
    main()
