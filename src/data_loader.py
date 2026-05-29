import os
import numpy as np
from scipy.integrate import solve_ivp
from physics import CislunarEnvironment 

class OrbitalDataLoader:
    def __init__(self):
        self.env = CislunarEnvironment()
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.current_dir)
        self.data_dir = os.path.join(self.project_root, 'data')
        os.makedirs(self.data_dir, exist_ok=True)

    def generate_spatial_trajectory(self):
        """
        Generates a flawless spatial (3D) periodic Halo orbit around L1.
        Uses high-precision RK45 integration.
        """
        # 1. Spatial Initial Conditions: L1 Northern Halo Orbit
        # These are pre-calculated to ensure stability. Notice non-zero Z position and Y velocity.
        X0_spatial = np.array([0.836915, 0.0, 0.150020, 0.0, 0.215033, 0.0])
        orbital_period_T = 2.743 # Normalized time for full loop
        
        print("\n--- Initiating Spatial Propagation (RK45) ---")
        
        # 2. Configure High-Precision RK45 Solver
        # We integrate from 0 to T for one complete orbital period.
        solution = solve_ivp(
            fun=self.env.equations_of_motion,
            t_span=(0, orbital_period_T),
            y0=X0_spatial,
            method='RK45', 
            atol=1e-12, # Extreme tolerances required for chaotic cislunar dynamics
            rtol=1e-10,
            dense_output=True # Crucial for Phase 5 extraction
        )
        
        if solution.success:
            print(f"Integration Successful. Trajectory mapped for full period T={orbital_period_T}")
            
            # 3. Dense Output Extraction (Phase 5 Logic)
            # Generate 10,000 perfectly spaced time points
            num_points = 10000
            t_eval = np.linspace(0, orbital_period_T, num_points)
            
            # Extract the flawless, uniform 6D trajectory array
            trajectory_raw = solution.sol(t_eval).T
            
            # 4. Feature Scaling (Preprocessing for Neural Network)
            # Min-Max Scaling compresses the variables to (-1, 1) bounds (cite: 1329-1331)
            def min_max_scale(data):
                data_min = np.min(data, axis=0)
                data_max = np.max(data, axis=0)
                
                range_vals = data_max - data_min
                # Handle edge case where variable is perfectly constant (rare in spatial)
                range_vals[range_vals == 0] = 1e-12 
                
                scaled_data = 2 * ((data - data_min) / range_vals) - 1
                return scaled_data, data_min, data_max

            scaled_trajectory, norm_min, norm_max = min_max_scale(trajectory_raw)
            
            # 5. Serialization: Saving both Raw and Scaled datasets
            scaled_save_path = os.path.join(self.data_dir, 'ground_truth_scaled.npy')
            raw_save_path = os.path.join(self.data_dir, 'ground_truth_raw.npy')
            norm_params_path = os.path.join(self.data_dir, 'normalization_params.npz')

            np.save(scaled_save_path, scaled_trajectory)
            np.save(raw_save_path, trajectory_raw)
            np.savez(norm_params_path, min=norm_min, max=norm_max) # Store scaling parameters

            print(f"Dataset generated. Saved Scaled ({scaled_trajectory.shape}) and Raw datasets to /data/")
            return trajectory_raw
            
        else:
            print(f"Integration Failed: {solution.message}")
            return None

if __name__ == "__main__":
    loader = OrbitalDataLoader()
    loader.generate_spatial_trajectory()