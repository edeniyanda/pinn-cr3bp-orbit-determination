import numpy as np

class CislunarEnvironment:
    """
    Standard mathematical model for the Spatial (3D) Circular Restricted Three-Body Problem (CR3BP).
    This environment handles the physics of a massless spacecraft navigating the barycentric
    rotating frame of two massive primaries (e.g., Earth and Moon).
    """
    def __init__(self, mu=0.0121505856):
        """
        Initializes the dynamic environment.
        Args:
            mu (float): The standardized, dimensionless mass ratio of the 
                       secondary body to the system total (M_M / (M_E + M_M)).
                       Defaults to the Earth-Moon system value (~0.01215).
                       This single parameter fully defines the normalized system.
        """
        # --- The System Mass Parameter ---
        self.mu = mu
        # Note: The normalized mass of the Earth (Primary) is (1 - mu).

        # --- Canonical Positions of Primaries ---
        # The coordinate system is barycentric (origin is center of mass).
        # Both primaries are fixed on the x-axis, separated by 1.0 normalized units.
        
        # Earth (M1) sits very close to the barycenter
        self.earth_pos_x = -self.mu
        
        # Moon (M2) sits at approximately 1 normalized unit
        self.moon_pos_x = 1.0 - self.mu

    def equations_of_motion(self, t, X):
        """
        The core physics engine for the Spatial CR3BP.
        Calculates the rates of change (derivatives) for the 6D state vector.
        
        Args:
            t (float): Normalized time. (Problem is time-independent).
            X (array-like): The 6D state vector in the rotating frame:
                             [x, y, z, vx, vy, vz].
        
        Returns:
            np.ndarray: [dx, dy, dz, dvx, dvy, dvz] (accelerations include Coriolis/Centrifugal)
        """
        # Unpack the current 6D state
        x, y, z, vx, vy, vz = X
        
        # 1. Calculate relative distances to both massive bodies in 3D space
        # d_earth^2 = (x + mu)^2 + y^2 + z^2
        r1_cubed = ((x - self.earth_pos_x)**2 + y**2 + z**2)**1.5
        
        # d_moon^2 = (x - (1-mu))^2 + y^2 + z^2
        r2_cubed = ((x - self.moon_pos_x)**2 + y**2 + z**2)**1.5
        
        # 2. Compute the spatial gradients of the pseudo-potential potential (forces)
        # Standard rotating potential: Omega = 1/2(x^2+y^2) + (1-mu)/r1 + mu/r2
        
        # Centrifugal + Gravitational forces
        dU_dx = x - ((1 - self.mu) * (x - self.earth_pos_x) / r1_cubed) - \
                   (self.mu * (x - self.moon_pos_x) / r2_cubed)
                   
        dU_dy = y - ((1 - self.mu) * y / r1_cubed) - (self.mu * y / r2_cubed)
        
        # Vertical gradient: No centripetal force along rotating axis (only gravity)
        dU_dz =   - ((1 - self.mu) * z / r1_cubed) - (self.mu * z / r2_cubed)
        
        # 3. Assemble accelerations including the Coriolis effect
        # accelerations: a_x = dU_dx + 2*vy
        #                a_y = dU_dy - 2*vx
        
        rates = [
            vx,            # rate of change of x (dx/dt)
            vy,            # rate of change of y (dy/dt)
            vz,            # rate of change of z (dz/dt)
            2*vy + dU_dx,  # Acceleration along X [a_x]
           -2*vx + dU_dy,  # Acceleration along Y [a_y]
            dU_dz          # Acceleration along Z [a_z]
        ]
        
        return np.array(rates)

    def calculate_jacobi_constant(self, X_spatial):
        """
        Calculates the Jacobi Constant (C), an energetic invariant in the rotating frame.
        Crucial requirement for your novel energy penalty loss function.
        
        Args:
            X_spatial (np.ndarray): Single 6D state vector or array of shape (N, 6).
            
        Returns:
            np.ndarray: The Jacobi Constant value(s) for the given state(s).
        """
        # Vectorized support for handling single states or batches (N, 6)
        if X_spatial.ndim == 1:
            x, y, z, vx, vy, vz = X_spatial
        else:
            x, y, z, vx, vy, vz = X_spatial.T
        
        # 1. Recalculate full 3D relative distances
        r1 = np.sqrt((x - self.earth_pos_x)**2 + y**2 + z**2)
        r2 = np.sqrt((x - self.moon_pos_x)**2 + y**2 + z**2)
        
        # 2. Compute the pseudo-potential term (Centrifugal + Gravitational Energy)
        # Omega = 1/2(x^2+y^2) + (1-mu)/r1 + mu/r2 + 1/2*mu*(1-mu)
        omega_term = (x**2 + y**2) + 2*(1 - self.mu)/r1 + 2*self.mu/r2 + self.mu*(1 - self.mu)
        
        # 3. Compute relative velocity squared
        velocity_sq = vx**2 + vy**2 + vz**2
        
        # C = Omega - V^2
        return omega_term - velocity_sq