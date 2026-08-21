import torch
import casadi as ca
import numpy as np

# =============================================================================
# PART 1: PYTORCH DIFFERENTIABLE CONSTRAINTS 
# Used for: Soft-loss training baseline & calculating Physics RMSE metrics
# =============================================================================

def compute_physics_residuals(x_scaled, y_scaled, scaler_X, scaler_Y):
    """
    Computes the normalized constraint residuals directly on PyTorch tensors.
    Fully differentiable and GPU-compatible.
    """
    device = x_scaled.device
    
    # Convert scalers to tensors on the correct device
    X_scale = torch.tensor(scaler_X.scale_, device=device, dtype=torch.float32)
    X_mean = torch.tensor(scaler_X.mean_, device=device, dtype=torch.float32)
    Y_scale = torch.tensor(scaler_Y.scale_, device=device, dtype=torch.float32)
    Y_mean = torch.tensor(scaler_Y.mean_, device=device, dtype=torch.float32)
    
    # 1. Unscale to Physical Units
    x_phys = x_scaled * X_scale + X_mean
    y_phys = y_scaled * Y_scale + Y_mean
    
    P = x_phys[:, 0]
    I = x_phys[:, 1]
    R = x_phys[:, 2]
    
    # 2. Extract Physical Variables
    Tg = y_phys[:, 11]            # gas temperature
    kb = 1.380649e-23
    e  = 1.602176634e-19
    
    conc_sum = y_phys[:, :11].sum(dim=1)  # sum of species 0 to 10
    P_calc = conc_sum * Tg * kb           # Ideal Gas Law
    
    ne_model = y_phys[:, 16]
    ne_calc = y_phys[:, 4] + y_phys[:, 7] - y_phys[:, 8] # Quasi-neutrality
    
    vd = y_phys[:, 14]
    I_calc = e * ne_model * vd * torch.pi * R**2         # Current conservation
    
    # 3. Compute Normalized Residuals (scaled back to feature domain for stable gradients)
    res_P = (P_calc - P) / X_scale[0]
    res_I = (I_calc - I) / X_scale[1]
    res_ne = (ne_calc - ne_model) / Y_scale[4]
    
    # Stack into shape (Batch, 3)
    return torch.stack([res_P, res_I, res_ne], dim=1)


# =============================================================================
# PART 2: CASADI HARD PROJECTION SOLVER
# Used for: Post-hoc projection of generative predictions onto the physical manifold
# =============================================================================

def define_g_constraint(scaler_X, scaler_Y, x_dim=3, p_dim=17):
    """Defines the symbolic non-linear programming (NLP) constraints for CasADi."""
    x_sym = ca.SX.sym('x', x_dim)
    p_sym = ca.SX.sym('p', p_dim)
    
    # Unscale Inputs
    P = x_sym[0] * scaler_X.scale_[0] + scaler_X.mean_[0]
    I = x_sym[1] * scaler_X.scale_[1] + scaler_X.mean_[1]
    R = x_sym[2] * scaler_X.scale_[2] + scaler_X.mean_[2]
    kb, e = 1.380649e-23, 1.602176634e-19
    
    # Unscale Targets
    Tg = p_sym[11] * scaler_Y.scale_[11] + scaler_Y.mean_[11]
    
    conc_scaled_sum = 0
    for i in range(11):
        conc_scaled_sum += (p_sym[i] * scaler_Y.scale_[i] + scaler_Y.mean_[i])
        
    P_calc = conc_scaled_sum * Tg * kb
    
    ne_model = p_sym[16] * scaler_Y.scale_[16] + scaler_Y.mean_[16]
    ne_calc = (p_sym[4] * scaler_Y.scale_[4] + scaler_Y.mean_[4] + 
               p_sym[7] * scaler_Y.scale_[7] + scaler_Y.mean_[7] - 
              (p_sym[8] * scaler_Y.scale_[8] + scaler_Y.mean_[8]))
    
    vd = p_sym[14] * scaler_Y.scale_[14] + scaler_Y.mean_[14]
    I_calc = e * ne_model * vd * ca.pi * R**2
    
    # Normalized Constraints
    P_law = (P_calc - P) / scaler_X.scale_[0]
    I_law = (I_calc - I) / scaler_X.scale_[1]
    ne_law = (ne_calc - ne_model) / scaler_Y.scale_[4]
    
    h = ca.vertcat(P_law, I_law, ne_law)
    return ca.Function('g_casadi', [x_sym, p_sym], [h])


class ProjectionSolver:
    """
    Solves the constrained optimization problem mapping raw predictions 
    to the strictly valid geometric physical manifold.
    """
    def __init__(self, scaler_X, scaler_Y, x_dim=3, p_dim=17):
        self.p_dim = p_dim
        g_casadi = define_g_constraint(scaler_X, scaler_Y, x_dim, p_dim)
        
        # Decision variable (the projected physically valid state)
        p_var = ca.SX.sym('p_var', p_dim)
        
        # Symbolic placeholders for the conditions (x) and the raw network prediction (y_hat)
        x_param = ca.SX.sym('x_param', x_dim)
        y_target_param = ca.SX.sym('y_target_param', p_dim)

        # Objective: Orthogonal L2 minimization || y_valid - y_hat ||^2
        objective = ca.sumsqr(p_var - y_target_param)
        constraints = g_casadi(x_param, p_var)
        
        nlp_problem = {
            'x': p_var,
            'f': objective,
            'g': constraints,
            'p': ca.vcat([x_param, y_target_param])
        }
        
        options = {
            'ipopt.print_level': 0, 'ipopt.sb': "no", 'ipopt.tol': 1e-10, 
            'print_time': 0, 'ipopt.max_iter': 500, 'ipopt.acceptable_tol': 1e-10
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_problem, options)
        
        # Equality constraints g(x,p) = 0
        self.lbg = [0] * constraints.shape[0]
        self.ubg = [0] * constraints.shape[0]
    
    def solve_batch(self, x_fixed_batch, y_pred_batch, S_mask_batch=None):
        """
        Maps a batch of raw numpy predictions to the physical manifold.
        Args:
            x_fixed_batch: Numpy array of shape (B, x_dim)
            y_pred_batch: Raw generative predictions (B, p_dim)
            S_mask_batch: Optional binary mask (B, p_dim). 1 = observed (fixed), 0 = unobserved (free)
        Returns:
            p_optimal_scaled_batch: Projected predictions (B, p_dim)
        """
        B = x_fixed_batch.shape[0]
        p_optimal_scaled_batch = np.zeros((B, self.p_dim), dtype=np.float32)
        
        for i in range(B):
            current_x = x_fixed_batch[i, :]
            current_y_pred = y_pred_batch[i, :]
            
            param_values = np.concatenate([current_x, current_y_pred])
            
            # Default to unbounded for all variables
            lbx = [-np.inf] * self.p_dim
            ubx = [np.inf] * self.p_dim
            
            # If a mask is provided, lock the observed variables exactly in place
            if S_mask_batch is not None:
                current_mask = S_mask_batch[i, :]
                for j in range(self.p_dim):
                    if current_mask[j] == 1.0:
                        lbx[j] = current_y_pred[j]
                        ubx[j] = current_y_pred[j]
            
            try:
                # x0 is initialized at the highly accurate generative prior
                sol = self.solver(x0=current_y_pred, 
                                p=param_values,
                                lbx=lbx,         # Apply Lower Bounds
                                ubx=ubx,         # Apply Upper Bounds
                                lbg=self.lbg, 
                                ubg=self.ubg)
                
                p_optimal_scaled_batch[i, :] = sol['x'].toarray().flatten()
            except Exception as e:
                # If solver fails, return NaNs so we don't silently ignore it
                print(f"Solver failed for item {i}: {e}")
                p_optimal_scaled_batch[i, :] = np.full(self.p_dim, np.nan)
        
        return p_optimal_scaled_batch