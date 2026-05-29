import os
import torch
import torch.nn as nn
from network import Jacobi_PINN




def calculate_physics_loss(pinn_model, t_tensor, mu, C_true):
    """
    Calculates the PDE residual loss and the Jacobi Constant penalty.
    t_tensor: A column tensor of times with requires_grad=True
    mu: The mass parameter (approx 0.01215)
    C_true: The exact Jacobi Constant calculated from your initial state X0
    """
    
    # Forward Pass: Get the constrained predictions
    preds = pinn_model(t_tensor)
    x  = preds[:, 0:1]
    y  = preds[:, 1:2]
    z  = preds[:, 2:3]
    vx = preds[:, 3:4]
    vy = preds[:, 4:5]
    vz = preds[:, 5:6]
    
    # Automatic Differentiation (Extracting Accelerations)
    # We take the derivative of the predicted velocities with respect to time (t)
    # create_graph=True allows us to backpropagate through this derivative later
    dvx_dt = torch.autograd.grad(vx, t_tensor, grad_outputs=torch.ones_like(vx), create_graph=True)[0]
    dvy_dt = torch.autograd.grad(vy, t_tensor, grad_outputs=torch.ones_like(vy), create_graph=True)[0]
    dvz_dt = torch.autograd.grad(vz, t_tensor, grad_outputs=torch.ones_like(vz), create_graph=True)[0]
    
    # Dynamic Geometry (Distances to Earth and Moon)
    r1 = torch.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = torch.sqrt((x - (1 - mu))**2 + y**2 + z**2)
    
    # otential Gradients (The Forces)
    dU_dx = x - ((1 - mu) * (x + mu) / r1**3) - (mu * (x - (1 - mu)) / r2**3)
    dU_dy = y - ((1 - mu) * y / r1**3) - (mu * y / r2**3)
    dU_dz =   - ((1 - mu) * z / r1**3) - (mu * z / r2**3)
    
    # The ODE Residuals (Equations of Motion)
    # If the network predicts perfect physics, these will all equal 0.0
    res_x = dvx_dt - 2*vy - dU_dx
    res_y = dvy_dt + 2*vx - dU_dy
    res_z = dvz_dt - dU_dz
    
    loss_ODE = torch.mean(res_x**2 + res_y**2 + res_z**2)
    
    # The Jacobi Constant Penalty (Your Thesis Contribution)
    # Calculate the instantaneous energy at every predicted coordinate
    omega_term = (x**2 + y**2) + 2*(1 - mu)/r1 + 2*mu/r2 + mu*(1 - mu)
    velocity_sq = vx**2 + vy**2 + vz**2
    C_pred = omega_term - velocity_sq
    
    loss_C = torch.mean((C_pred - C_true)**2)
    
    return loss_ODE, loss_C



# Initialization
mu = 0.0121505856
X0 = [0.836915, 0.0, 0.150020, 0.0, 0.215033, 0.0] # Your perfected starting state
C_true = 3.189 # Calculate this analytically from X0 beforehand

# Instantiate your model and optimizer
model = Jacobi_PINN(X0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Prepare the time tensor (MUST have requires_grad=True for autograd to work)
t_train = torch.linspace(0, 2.74, 1000).view(-1, 1).requires_grad_(True)

# The Training Loop

# --- STAGE 1: Adam Optimization (Exploration) ---
print("--- Starting Stage 1: Adam Optimizer ---")
adam_epochs = 2000
for epoch in range(adam_epochs):
    optimizer.zero_grad()
    
    loss_ODE, loss_C = calculate_physics_loss(model, t_train, mu, C_true)
    
    w_ODE, w_C = 1.0, 10.0
    total_loss = (w_ODE * loss_ODE) + (w_C * loss_C)
    
    total_loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Adam Epoch {epoch} | ODE Loss: {loss_ODE.item():.6f} | Jacobi Loss: {loss_C.item():.6f}")

# --- STAGE 2: L-BFGS Optimization (Exploitation) ---
print("\n--- Starting Stage 2: L-BFGS Optimizer ---")
# L-BFGS requires strict parameter tuning to prevent it from stalling
lbfgs_optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=5000,
    max_eval=5000,
    tolerance_grad=1e-7,
    tolerance_change=1e-9,
    history_size=100,
    line_search_fn="strong_wolfe" # Critical for chaotic PDEs
)

# L-BFGS requires a closure function to re-evaluate the loss multiple times per step
epoch_lbfgs = 0
def closure():
    global epoch_lbfgs
    lbfgs_optimizer.zero_grad()
    
    loss_ODE, loss_C = calculate_physics_loss(model, t_train, mu, C_true)
    
    w_ODE, w_C = 1.0, 10.0
    total_loss = (w_ODE * loss_ODE) + (w_C * loss_C)
    
    total_loss.backward()
    
    if epoch_lbfgs % 100 == 0:
        print(f"L-BFGS Step {epoch_lbfgs} | ODE Loss: {loss_ODE.item():.8f} | Jacobi Loss: {loss_C.item():.8f}")
    
    epoch_lbfgs += 1
    return total_loss

# Execute the L-BFGS closure loop
lbfgs_optimizer.step(closure)

# --- SAVING THE MODEL ---
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
save_dir = os.path.join(project_root, 'models')
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, 'jacobi_pinn_weights.pth')
torch.save(model.state_dict(), save_path)
print(f"\nTraining complete. Model weights saved strictly to {save_path}")


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

