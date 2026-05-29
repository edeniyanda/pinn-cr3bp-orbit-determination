import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from network import Jacobi_PINN

def evaluate_model():
    print("\n--- Initiating Model Evaluation ---")

    # System Constants (From Phase 1 & 2)
    T = 2.743
    X0 = [0.836915, 0.0, 0.150020, 0.0, 0.215033, 0.0]

    # Use paths relative to the project root
    data_path = 'data/ground_truth_trajectory.npy'
    model_path = 'models/jacobi_pinn_weights.pth'

    # Load Ground Truth Data (SciPy RK45)
    try:
        trajectory_data = np.load(data_path)
        print(f"Classical ground truth loaded successfully. Shape: {trajectory_data.shape}")
    except FileNotFoundError:
        print(f"Error: {data_path} not found.")
        print("Please run 'python src/data_loader.py' first to generate the dataset.")
        return

    # Initialize Model and Load Weights
    model = Jacobi_PINN(X0)
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval() # Freeze weights for inference
        print(f"Pre-trained PINN weights loaded successfully from {model_path}.")
    except FileNotFoundError:
        print(f"Error: {model_path} not found.")
        print("Please run 'python src/train.py' first to train the network.")
        return

    # Perform Instantaneous Inference
    # generate a perfectly spaced time tensor and run the forward pass
    nuoints = 10000
    t_eval_tensor = torch.linspace(0, T, num_points).view(-1, 1)

    with torch.no_grad():
        predicted_states = model(t_eval_tensor).numpy()

    # Calculate Metrics (Spatial MSE)
    # slice [:, 0:3] to isolate x, y, z coordinates and ignore velocity for the plot
    ground_truth_spatial = trajectory_data[:, 0:3]
    predicted_spatial = predicted_states[:, 0:3]
    
    mse_spatial = np.mean((ground_truth_spatial - predicted_spatial)**2)
    print(f"\n>>> Mathematical Deviation (Spatial MSE): {mse_spatial:.6f} <<<")

    # Visual Manifold Mapping (Minimalist Dark Mode)
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Classical Truth
    ax.plot(trajectory_data[:, 0], trajectory_data[:, 1], trajectory_data[:, 2], 
            color='#FFFFFF', linewidth=2, linestyle='--', label='Classical Truth (RK45)')
    
    # Plot PINN Prediction
    ax.plot(predicted_states[:, 0], predicted_states[:, 1], predicted_states[:, 2], 
            color='#00FF41', linewidth=1.5, label='PINN Prediction')

    # Plot Canonical Primaries
    ax.scatter([-0.01215], [0], [0], color='#0078D7', s=100, label='Earth')
    ax.scatter([0.98785], [0], [0], color='#AAAAAA', s=50, label='Moon')

    # Formatting and UI
    ax.set_title(f'Trajectory Comparison | Spatial MSE: {mse_spatial:.4f}', color='white', pad=20)
    ax.set_xlabel('X (Canonical)', color='gray')
    ax.set_ylabel('Y (Canonical)', color='gray')
    ax.set_zlabel('Z (Canonical)', color='gray')
    
    ax.grid(color='#333333', linestyle='--', linewidth=0.5)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.legend(frameon=False, labelcolor='white')

    plt.show()

if __name__ == "__main__":
    # Ensure necessary directories exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    evaluate_model()
