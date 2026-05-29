import os
import numpy as np
import pandas as pd

def inspect_terminal_output():
    # Load the raw, unscaled physics data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data', 'ground_truth_raw.npy')
    
    try:
        raw_data = np.load(data_path)
    except FileNotFoundError:
        print("Run data_loader.py first to generate the raw dataset.")
        return

    # Convert to a Pandas DataFrame for beautiful terminal formatting
    columns = ['X (Pos)', 'Y (Pos)', 'Z (Pos)', 'Vx (Vel)', 'Vy (Vel)', 'Vz (Vel)']
    df = pd.DataFrame(raw_data, columns=columns)

    # Terminal Output Readout
    print("\n" + "="*70)
    print("🚀 CISLUNAR ORBITAL DYNAMICS: TERMINAL READOUT")
    print("="*70)
    
    print("\n--- Orbit Diagnostics ---")
    z_max = df['Z (Pos)'].max()
    z_min = df['Z (Pos)'].min()
    
    if abs(z_max - z_min) < 1e-10:
        print("Orbit Type: PLANAR (2D)")
        print("Confirmation: Z-axis remains flat at 0.0 across all time steps.")
    else:
        print("Orbit Type: SPATIAL (3D)")
        print(f"Confirmation: Z-axis fluctuates between {z_min:.4f} and {z_max:.4f}.")

    print("\n--- First 5 Microsecond Time Steps ---")
    print(df.head(5).to_string(index=True))
    
    print("\n--- Final 5 Microsecond Time Steps (Loop Closure) ---")
    print(df.tail(5).to_string(index=True))
    print("="*70 + "\n")

if __name__ == "__main__":
    inspect_terminal_output()