import os
import torch
import numpy as np
import torch.nn as nn
from network import Jacobi_PINN
from physics import CislunarEnvironment

def calculate_physics_loss(model, t_tensor, env, C_true):
    """
    Calculates the 3D PDE residual loss and your Jacobi Constant penalty.
    """
    # 1. Forward Pass
    preds = model(t_tensor)
    x, y, z = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3]
    vx, vy, vz = preds[:, 3:4], preds[:, 4:5], preds[:, 5:6]
    
    # 2. Extract Gradients via Autograd (Kinematics & Accelerations)
    dx_dt = torch.autograd.grad(x, t_tensor, torch.ones_like(x), create_graph=True)[0]
    dy_dt = torch.autograd.grad(y, t_tensor, torch.ones_like(y), create_graph=True)[0]
    dz_dt = torch.autograd.grad(z, t_tensor, torch.ones_like(z), create_graph=True)[0]
    
    dvx_dt = torch.autograd.grad(vx, t_tensor, torch.ones_like(vx), create_graph=True)[0]
    dvy_dt = torch.autograd.grad(vy, t_tensor, torch.ones_like(vy), create_graph=True)[0]
    dvz_dt = torch.autograd.grad(vz, t_tensor, torch.ones_like(vz), create_graph=True)[0]
    
    # 3. Dynamic Geometry & Pseudo-Potential Gradients
    r1_cubed = ((x - env.earth_pos_x)**2 + y**2 + z**2)**1.5
    r2_cubed = ((x - env.moon_pos_x)**2 + y**2 + z**2)**1.5
    
    dU_dx = x - ((1 - env.mu)*(x - env.earth_pos_x)/r1_cubed) - (env.mu*(x - env.moon_pos_x)/r2_cubed)
    dU_dy = y - ((1 - env.mu)*y/r1_cubed) - (env.mu*y/r2_cubed)
    dU_dz =   - ((1 - env.mu)*z/r1_cubed) - (env.mu*z/r2_cubed)
    
    # 4. ODE Residuals (Forces Network to obey Newton/Coriolis)
    res_x = dx_dt - vx
    res_y = dy_dt - vy
    res_z = dz_dt - vz
    res_vx = dvx_dt - (2*vy + dU_dx)
    res_vy = dvy_dt - (-2*vx + dU_dy)
    res_vz = dvz_dt - dU_dz
    
    loss_ODE = torch.mean(res_x**2 + res_y**2 + res_z**2 + res_vx**2 + res_vy**2 + res_vz**2)


    # 5. Jacobi Constant Penalty (Enforces Zero Velocity Surfaces)
    r1 = torch.sqrt((x - env.earth_pos_x)**2 + y**2 + z**2)
    r2 = torch.sqrt((x - env.moon_pos_x)**2 + y**2 + z**2)
    omega = (x**2 + y**2) + 2*(1 - env.mu)/r1 + 2*env.mu/r2 + env.mu*(1 - env.mu)
    v_sq = vx**2 + vy**2 + vz**2
    C_pred = omega - v_sq
    
    loss_C = torch.mean((C_pred - C_true)**2)
    
    # REMOVED L_p ENTIRELY
    return loss_ODE, loss_C

def train_model():
    # Environment & Initialization
    env = CislunarEnvironment()
    X0 = [0.836915, 0.0, 0.150020, 0.0, 0.215033, 0.0]
    T = 2.743
    
    # 1. LOAD GROUND TRUTH DATA (The Missing Anchor)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data', 'ground_truth_raw.npy')
    
    trajectory_data = np.load(data_path)
    # We generated 10000 points in data_loader, we will train on them
    X_true = torch.tensor(trajectory_data, dtype=torch.float32)
    
    # Time tensor matching the data points
    num_points = 10000
    t_train = torch.linspace(0, T, num_points).view(-1, 1).requires_grad_(True)

    C_true = env.calculate_jacobi_constant(np.array(X0))
    C_true = torch.tensor(C_true, dtype=torch.float32)

    model = Jacobi_PINN(X0)

    # --- STAGE 1: Adam Optimizer ---
    print("--- Starting Stage 1: Adam Optimizer ---")
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 2000
    
    for epoch in range(epochs):
        adam_optimizer.zero_grad()
        
        # Get physics loss
        loss_ODE, loss_C = calculate_physics_loss(model, t_train, env, C_true)
        
        # Get Data Loss (MSE against SciPy truth)
        preds = model(t_train)
        loss_Data = torch.mean((preds - X_true)**2)
        
        # HYBRID TOTAL LOSS: Data + Physics + Energy Penalty
        total_loss = (1.0 * loss_Data) + (1e-4 * loss_ODE) + (1e-3 * loss_C)
        
        total_loss.backward()
        adam_optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Adam Epoch {epoch} | Data: {loss_Data.item():.5f} | ODE: {loss_ODE.item():.5f} | Jacobi: {loss_C.item():.5f}")
            
    # --- STAGE 2: L-BFGS Optimizer ---
    print("\n--- Starting Stage 2: L-BFGS Optimizer ---")
    lbfgs_optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=5000, tolerance_grad=1e-7,
        tolerance_change=1e-9, history_size=100, line_search_fn="strong_wolfe"
    )

    epoch_lbfgs = 0
    def closure():
        nonlocal epoch_lbfgs
        lbfgs_optimizer.zero_grad()
        
        loss_ODE, loss_C = calculate_physics_loss(model, t_train, env, C_true)
        preds = model(t_train)
        loss_Data = torch.mean((preds - X_true)**2)
        
        total_loss = (1.0 * loss_Data) + (1e-4 * loss_ODE) + (1e-3 * loss_C)
        total_loss.backward()
        
        if epoch_lbfgs % 100 == 0:
            print(f"L-BFGS Step {epoch_lbfgs} | Data: {loss_Data.item():.6f} | ODE: {loss_ODE.item():.6f}")
        epoch_lbfgs += 1
        return total_loss

    lbfgs_optimizer.step(closure)
    
    # --- SAVE MODEL ---
    save_dir = os.path.join(project_root, 'models')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'jacobi_pinn_weights.pth')
    
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete. Model weights saved to {save_path}")

    
if __name__ == "__main__":
    train_model()

# for epoch in range(epochs):
#     optimizer.zero_grad()
    
#     # Calculate the dual-objective loss
#     loss_ODE, loss_C = calculate_physics_loss(model, t_train, mu, C_true)
    
#     # Weighted Total Loss (You can tune these weights)
#     w_ODE, w_C = 1.0, 10.0 # Heavy penalty on the Jacobi boundary
#     total_loss = (w_ODE * loss_ODE) + (w_C * loss_C)
    
#     total_loss.backward()
#     optimizer.step()
    
#     if epoch % 500 == 0:
#         print(f"Epoch {epoch} | ODE Loss: {loss_ODE.item():.6f} | Jacobi Loss: {loss_C.item():.6f}")


# current_dir = os.path.dirname(os.path.abspath(__file__))

# project_root = os.path.dirname(current_dir)


# save_dir = os.path.join(project_root, 'models')


# os.makedirs(save_dir, exist_ok=True)

# # Define the full path and save
# save_path = os.path.join(save_dir, 'jacobi_pinn_weights.pth')
# torch.save(model.state_dict(), save_path)

# print(f"Training complete. Model weights saved strictly to {save_path}")

