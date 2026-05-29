import torch
import torch.nn as nn

class Jacobi_PINN(nn.Module):
    def __init__(self, X0):
        super(Jacobi_PINN, self).__init__()
        
        # Store the perfect Spatial initial boundary
        self.X0 = torch.tensor(X0, dtype=torch.float32)
        
        # Deep Learning Topology (6 inputs for 3D state)
        self.network = nn.Sequential(
            nn.Linear(1, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 6) # Outputs [x, y, z, vx, vy, vz]
        )
        

    def forward(self, t):
        T = 2.743
        
        # Normalize time to [-1, 1] for stable Tanh activation
        t_norm = (t / T) * 2.0 - 1.0 
        
        raw_output = self.network(t_norm)
        raw_x, raw_y, raw_z, raw_vx, raw_vy, raw_vz = raw_output.chunk(6, dim=1)
        
        # Keep the existing time_scale for the hard constraints
        time_scale = t / T 
        
        x  = time_scale * raw_x  + self.X0[0]
        y  = time_scale * raw_y  + self.X0[1]
        z  = time_scale * raw_z  + self.X0[2]
        vx = time_scale * raw_vx + self.X0[3]
        vy = time_scale * raw_vy + self.X0[4] 
        vz = time_scale * raw_vz + self.X0[5]
        
        return torch.cat([x, y, z, vx, vy, vz], dim=1)