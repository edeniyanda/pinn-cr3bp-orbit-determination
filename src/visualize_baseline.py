import os
import numpy as np
import matplotlib.pyplot as plt

def plot_classical_baseline():
    print("--- Loading Classical Ground Truth ---")
    
    # Resolve the absolute path to your data file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data', 'ground_truth_raw.npy')
    
    try:
        # Load the unscaled SciPy ground truth data
        trajectory_data = np.load(data_path)
        print(f"Data loaded successfully. Shape: {trajectory_data.shape}")
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}.")
        print("Ensure you have run data_loader.py to generate the orbit.")
        return

    # Extract spatial coordinates (ignoring velocities for the plot)
    x = trajectory_data[:, 0]
    y = trajectory_data[:, 1]
    z = trajectory_data[:, 2]

    # Set up the visualization environment
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the classical trajectory
    ax.plot(x, y, z, color='#FFFFFF', linewidth=2, label='Classical Orbit (SciPy RK45)')

    # Plot the primary bodies (Earth and Moon)
    # Using the standard CR3BP canonical coordinates (mu ~ 0.01215)
    mu = 0.01215
    ax.scatter([-mu], [0], [0], color='#0078D7', s=100, label='Earth')
    ax.scatter([1 - mu], [0], [0], color='#AAAAAA', s=50, label='Moon')

    # Format the plot for clear spatial reading
    ax.set_title('Classical CR3BP Ground Truth Trajectory', color='white', pad=20)
    ax.set_xlabel('X (Canonical)', color='gray')
    ax.set_ylabel('Y (Canonical)', color='gray')
    ax.set_zlabel('Z (Canonical)', color='gray')

    # Clean up the grid axes
    ax.grid(color='#333333', linestyle='--', linewidth=0.5)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.legend(frameon=False, labelcolor='white')

    # Render the 3D plot
    plt.show()

if __name__ == "__main__":
    plot_classical_baseline()