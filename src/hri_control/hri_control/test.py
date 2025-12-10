import rclpy
from hri_control.hri_env_final import HriEnv
from stable_baselines3 import SAC
import time

MODEL_PATH = "sac_hri_final_final.zip" 

def main():
    rclpy.init()
    print("TESTING SAC")
    env = HriEnv() #ros publishers nd subscribers talking to gazebo
    print(f"Loading model from {MODEL_PATH}")
    try:
        model = SAC.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"ERROR Could not find model at {MODEL_PATH}")
        return

    print("Model loaded successfully.")
    #moving t start pos and sensor reading
    obs, _ = env.reset()
    step = 0
    #control loop
    while rclpy.ok():
        action, _ = model.predict(obs, deterministic=True)#ppick the best action #no random exploration 
        obs, reward, terminated, truncated, info = env.step(action)#new sensor readings,reaward for last move,end?,forceend
        dist = info.get('dist', 0.0)
        print(f"[Step {step}] Dist: {dist:.3f} | Reward: {reward:.3f}")
        
        step += 1

        if terminated or truncated:
            print("\n--- Episode finished ---\n")
            obs, _ = env.reset()
            step = 0
            time.sleep(1.0) 

    env.close()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
