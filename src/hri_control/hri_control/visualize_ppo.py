# visualize_ppo.py

import rclpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from hri_control.hri_env_final import HriEnv
from stable_baselines3 import PPO  # <--- Using PPO
import numpy as np

MODEL_PATH = "ppo_hri_final.zip"

def main():
    rclpy.init()
    env = HriEnv()
    
    print(f"Loading PPO Model: {MODEL_PATH}")
    try:
        model = PPO.load(MODEL_PATH)
    except FileNotFoundError:
        print("Model not found!")
        return

    # Run Episode
    obs, _ = env.reset()
    robot_path = []
    target_path = []
    distances = []
    
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        
        # Capture Data
        ee_state = env.fk.compute_fk(env.last_pos)
        robot_path.append(ee_state[:3])
        target_path.append(env.last_target[:3])
        distances.append(np.linalg.norm(ee_state[:3] - env.last_target[:3]))

        if term or trunc: break

    env.close()
    rclpy.shutdown()

    # Plotting
    robot_path = np.array(robot_path)
    target_path = np.array(target_path)
    
    fig = plt.figure(figsize=(14, 6))

    # 3D Trajectory
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(robot_path[:,0], robot_path[:,1], robot_path[:,2], color='orange', linewidth=2, label='PPO Path')
    ax1.scatter(target_path[-1,0], target_path[-1,1], target_path[-1,2], color='red', marker='x', s=200)
    ax1.set_title('PPO Trajectory (Baseline)')
    ax1.legend()

    # Distance
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(distances, color='orange', linewidth=2, label='PPO Distance')
    ax2.axhline(y=0.05, color='green', linestyle='--', label='Goal (5cm)')
    ax2.set_title('PPO Convergence Speed')
    ax2.legend()
    
    plt.show()

if __name__ == "__main__":
    main()
