import numpy as np
from scipy.integrate import solve_ivp


class CislunarEnvironment:
    def __init__(self):
        # Absolute SI Constants
        self.G = 6.67430e-11        # m^3 / (kg s^2)
        self.M_E = 5.9722e24        # kg
        self.M_M = 7.3477e22        # kg
        self.R_EM = 3.844e8         # meters

        # Characteristic Scaling Factors
        self.l_star = self.R_EM
        self.m_star = self.M_E + self.M_M
        self.t_star = np.sqrt((self.l_star**3) / (self.G * self.m_star))
        
        # Velocity scalar (useful later for converting the final outputs back to m/s)
        self.v_star = self.l_star / self.t_star

        # The Dimensionless Mass Ratio
        self.mu = self.M_M / self.m_star
        # The Earth's normalized mass is simply (1 - mu)

        # Canonical Coordinates of Primaries
        # Earth is heavy, so it sits very close to the 0,0,0 origin
        self.earth_pos = np.array([-self.mu, 0.0, 0.0])
        
        # Moon is lighter, so it sits further out at almost 1.0
        self.moon_pos = np.array([1.0 - self.mu, 0.0, 0.0])



class DifferentialCorrector:
    def __init__(self, environment):
        self.env = environment
        self.tolerance = 1e-12

    def shoot_and_correct(self, initial_guess):
        # initial_guess = [x0, 0, z0, 0, vy0, 0]
        current_state = np.copy(initial_guess)
        error = 1.0
        
        # Loop until the symmetry error is microscopic
        while error > self.tolerance:
            
            # Integrate the orbit and the STM to the half-period (y crossing)
            final_state, stm_final = self._integrate_to_plane_crossing(current_state)
            
            # Extract the velocities that are supposed to be zero
            vx_f = final_state[3]
            vz_f = final_state[5]
            
            error_vector = np.array([vx_f, vz_f])
            error = np.linalg.norm(error_vector)
            
            if error <= self.tolerance:
                print("Orbit Converged!")
                return current_state # This is your perfect Ground Truth starting line
                
            # Extract the 2 x 2 Jacobian from the 6 x 6 STM
            # (Maps how changing  z0 and vy0 impacts vxf and vzf)
            jacobian = self._extract_jacobian(stm_final, final_state)
            
            # Newton-Raphson Update
            correction = np.linalg.inv(jacobian) @ error_vector
            
            # Apply the correction to z0 (index 2) and vy0 (index 4)
            current_state[2] -= correction[0]
            current_state[4] -= correction[1]
            
        return current_state