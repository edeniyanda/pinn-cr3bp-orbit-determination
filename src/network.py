import torch
import torch.nn as nn

class Jacobi_PINN(nn.Module):
    def __init__(self, X0):
        super(Jacobi_PINN, self).__init__()
        
        # Store the perfect initial boundary (No epsilon parameters needed!)
        self.X0 = torch.tensor(X0, dtype=torch.float32)
        
        #  Deep Learning Topology
        self.network = nn.Sequential(
            nn.Linear(1, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 6) # Outputs 6 raw state variables
        )
        
    def forward(self, t):
        """
        t: A column tensor of time steps.
        """
        raw_output = self.network(t)
        raw_x, raw_y, raw_z, raw_vx, raw_vy, raw_vz = raw_output.chunk(6, dim=1)
        
        # Hard Constraint (The Ansatz)
        x  = t * raw_x  + self.X0[0]
        y  = t * raw_y  + self.X0[1]
        z  = t * raw_z  + self.X0[2]
        vx = t * raw_vx + self.X0[3]
        vy = t * raw_vy + self.X0[4] 
        vz = t * raw_vz + self.X0[5]
        
        return torch.cat([x, y, z, vx, vy, vz], dim=1)
    

