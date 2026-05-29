import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from network import Jacobi_PINN
from physics import CislunarEnvironment

def evaluate_model():
    print("\n--- Initiating 3D Model Evaluation ---")
    
    T = 2.743
    X0 = [0.836915, 0.0, 0.150020, 0.0, 0.215033, 0.0]
    env = CislunarEnvironment()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data', 'ground_truth_raw.npy')
    model_path = os.path.join(project_root, 'models', 'jacobi_pinn_weights.pth')

    # Load SciPy Ground Truth
    try:
        trajectory_data = np.load(data_path)
    except FileNotFoundError:
        print("Error: Run data_loader.py first.")
        return

    # Load Model
    model = Jacobi_PINN(X0)
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
    except FileNotFoundError:
        print("Error: Run train.py first.")
        return

    # Inference
    num_points = 10000
    t_eval_tensor = torch.linspace(0, T, num_points).view(-1, 1)

    with torch.no_grad():
        predicted_states = model(t_eval_tensor).numpy()

    # Calculate Error
    ground_truth_spatial = trajectory_data[:, 0:3]
    predicted_spatial = predicted_states[:, 0:3]
    
    mse_spatial = np.mean((ground_truth_spatial - predicted_spatial)**2)
    print(f"\n>>> Mathematical Deviation (Spatial MSE): {mse_spatial:.8f} <<<")

    # 3D Visual Manifold Mapping
    # Minimalist dark mode configuration for high-contrast presentation
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the Classical Truth (SciPy) - Subtle white dashed line
    ax.plot(trajectory_data[:, 0], trajectory_data[:, 1], trajectory_data[:, 2], 
            color='#FFFFFF', linewidth=1.5, linestyle='--', alpha=0.7, label='Classical Truth (RK45)')
    
    # Plot the PINN Prediction - Solid vibrant green line
    ax.plot(predicted_states[:, 0], predicted_states[:, 1], predicted_states[:, 2], 
            color='#00FF41', linewidth=2.0, label='PINN Prediction')

    # Plot Canonical Primaries
    # Earth (Blue) and Moon (Gray), sized relatively
    ax.scatter([env.earth_pos_x], [0], [0], color='#0078D7', s=150, edgecolors='white', linewidth=0.5, label='Earth')
    ax.scatter([env.moon_pos_x], [0], [0], color='#AAAAAA', s=60, edgecolors='white', linewidth=0.5, label='Moon')

    # Indicate the Start Position (t=0)
    # Using a high-contrast crimson marker with a white border to make it pop
    start_x, start_y, start_z = trajectory_data[0, 0], trajectory_data[0, 1], trajectory_data[0, 2]
    ax.scatter([start_x], [start_y], [start_z], 
               color='#FF3366', s=120, edgecolors='#FFFFFF', linewidth=1.5, zorder=5, label='Start Position (t=0)')

    # Minimalist Formatting and Typography
    ax.set_title(f'Spatial Halo Orbit | Spatial MSE: {mse_spatial:.6f}', color='white', pad=30, fontsize=14, fontweight='bold')
    
    ax.set_xlabel('\nX (Canonical)', color='#888888', fontsize=10)
    ax.set_ylabel('\nY (Canonical)', color='#888888', fontsize=10)
    ax.set_zlabel('\nZ (Canonical)', color='#888888', fontsize=10)
    
    # Clean up the background panes to create a floating void effect
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')

    # Subtle, non-distracting grid lines
    ax.grid(color='#222222', linestyle='-', linewidth=0.5)

    # Clean legend placement
    ax.legend(frameon=False, labelcolor='white', loc='upper right', fontsize=11)

    # Optimize layout margins
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_model()