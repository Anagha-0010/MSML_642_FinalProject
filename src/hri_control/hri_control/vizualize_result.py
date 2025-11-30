# visualize_results.py

import rclpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from hri_control.hri_env_final import HriEnv
from stable_baselines3 import SAC
import numpy as np

# --- CONFIGURATION ---
MODEL_PATH = "sac_hri_final_fresh.zip"  # Ensure this matches your file name
# ---------------------

def main():
    rclpy.init()
    
    # 1. Setup Environment & Load Model
    print("Setting up environment...")
    env = HriEnv()
    
    print(f"Loading model: {MODEL_PATH}")
    try:
        model = SAC.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH}' not found!")
        return

    # 2. Run ONE Test Episode
    print("Collecting trajectory data...")
    obs, _ = env.reset()
    
    robot_positions = [] # [x, y, z]
    target_positions = [] # [x, y, z]
    distances = []       # Scalar distance
    
    for step in range(100): # Run for max 100 steps
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Capture Data
        # env.fk.compute_fk returns [x, y, z, qx, qy, qz, qw]
        ee_state = env.fk.compute_fk(env.last_pos)
        robot_pos = ee_state[:3]
        target_pos = env.last_target[:3]
        
        # Calculate distance for the graph
        dist = np.linalg.norm(robot_pos - target_pos)
        
        robot_positions.append(robot_pos)
        target_positions.append(target_pos)
        distances.append(dist)

        if terminated or truncated:
            print(f"Goal reached or episode ended at step {step}")
            break

    env.close()
    rclpy.shutdown()

    # 3. Plotting
    print("Generating graphs...")
    robot_positions = np.array(robot_positions)
    target_positions = np.array(target_positions)
    steps = range(len(distances))

    fig = plt.figure(figsize=(14, 6))

    # --- PLOT 1: 3D Trajectory (Spatial) ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    # Plot Robot Path
    ax1.plot(robot_positions[:, 0], robot_positions[:, 1], robot_positions[:, 2], 
             label='Robot Wrist', color='blue', linewidth=2)
    
    # Plot Start Point
    ax1.scatter(robot_positions[0, 0], robot_positions[0, 1], robot_positions[0, 2], 
                color='green', s=100, label='Start')
    
    # Plot End Point
    ax1.scatter(robot_positions[-1, 0], robot_positions[-1, 1], robot_positions[-1, 2], 
                color='blue', s=100, label='End')

    # Plot Target Hand (Use the final position as the reference)
    ax1.scatter(target_positions[-1, 0], target_positions[-1, 1], target_positions[-1, 2], 
                color='red', marker='x', s=200, label='Target Hand')

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Handover Trajectory')
    ax1.legend()

    # --- PLOT 2: Distance vs. Time (Quantitative) ---
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(steps, distances, color='purple', linewidth=2, label='Distance to Hand')
    
    # Add a "Success Zone" line
    ax2.axhline(y=0.05, color='green', linestyle='--', label='Touch Threshold (5cm)')
    
    ax2.set_xlabel('Simulation Steps')
    ax2.set_ylabel('Distance (meters)')
    ax2.set_title('Convergence: How Fast Did It Reach?')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
