import torch
import numpy as np

# =============================================================================
# GENERALIZED SCHEDULE FUNCTIONS
# =============================================================================
def alpha_fn(t, schedule_type='quad'):
    if schedule_type == 'linear': return 1.0 - t
    elif schedule_type == 'quad': return (1.0 - t) ** 2
    raise ValueError(f"Unknown schedule: {schedule_type}")

def beta_fn(t, schedule_type='quad'):
    if schedule_type == 'linear': return t
    elif schedule_type == 'quad': return 1.0 - ((1.0 - t) ** 2)
    raise ValueError(f"Unknown schedule: {schedule_type}")

def dalpha_dt_fn(t, schedule_type='quad'):
    if schedule_type == 'linear': return -torch.ones_like(t)
    elif schedule_type == 'quad': return -2.0 * (1.0 - t)
    raise ValueError(f"Unknown schedule: {schedule_type}")

def dbeta_dt_fn(t, schedule_type='quad'):
    if schedule_type == 'linear': return torch.ones_like(t)
    elif schedule_type == 'quad': return 2.0 * (1.0 - t)
    raise ValueError(f"Unknown schedule: {schedule_type}")

def sigma_fn(t, sigma_max, schedule_type='quad'):
    return sigma_max * alpha_fn(t, schedule_type)

def dsigma_dt_fn(t, sigma_max, schedule_type='quad'):
    return sigma_max * dalpha_dt_fn(t, schedule_type)

# =============================================================================
# EXPLICIT TIME-DEPENDENT FLOW (CFM-t)
# =============================================================================
def inference_cfm_t(model, x, sigma_max, N_steps=100, schedule_type='quad', 
                    solver='heun', S=None, y_obs=None, pred_target='y'):
    """Integrates the continuous-time flow using explicit time steps."""
    batch_size = x.size(0)
    
    # ========================================================================
    # ENFORCE OT FLOW MATCHING BASELINE RULES
    # ========================================================================
    if pred_target == 'ot':
        S = torch.zeros(batch_size, model.y_dim, device=x.device)
        y_obs = torch.zeros(batch_size, model.y_dim, device=x.device)
        schedule_type = 'linear'
        sigma_max = 1.0
    elif S is None:
        S = torch.zeros(batch_size, model.y_dim, device=x.device)
        y_obs = torch.zeros(batch_size, model.y_dim, device=x.device)
        
    sig_0 = sigma_fn(torch.zeros(1, device=x.device), sigma_max, schedule_type).item()
    
    # 1. Initialize global isotropic noise (FIXED EPSILON)
    epsilon = torch.randn(batch_size, model.y_dim, device=x.device)
    
    # For OT, S=0, so this collapses cleanly to: I_t = sig_0 * epsilon
    I_t = S * (y_obs + sig_0 * epsilon) + (1 - S) * (sig_0 * epsilon)
    
    dt = 1.0 / N_steps
    
    for i in range(N_steps):
        t_tensor = torch.full((batch_size, 1), i * dt, device=x.device)
        
        with torch.no_grad():
            model_out, _ = model(x, I_t, t_tensor, S)
        
        # Calculate Vector Field based on exact generalized interpolant
        if pred_target == 'y':
            y_hat = model_out
            beta_t = beta_fn(t_tensor, schedule_type)
            dbeta_t = dbeta_dt_fn(t_tensor, schedule_type)
            sig_t = sigma_fn(t_tensor, sigma_max, schedule_type)
            dsig_t = dsigma_dt_fn(t_tensor, sigma_max, schedule_type)
            
            ratio = torch.where(sig_t > 1e-7, dsig_t / sig_t, torch.zeros_like(sig_t))
            y_coeff = S + (1 - S) * beta_t
            
            u_t = (1 - S) * dbeta_t * y_hat + ratio * (I_t - y_coeff * y_hat)
            
        elif pred_target in ['v', 'ot']:
            u_t = model_out
        else:
            raise ValueError(f"Unknown pred_target: {pred_target}")
        
        # Pre-calculate next time step parameters for harmonization
        t_next_tensor = torch.full((batch_size, 1), min((i + 1) * dt, 1.0), device=x.device)
        sig_next = sigma_fn(t_next_tensor, sigma_max, schedule_type)
        
        # ODE Solver Step
        if solver == 'euler' or i == N_steps - 1:
            I_raw = I_t + u_t * dt
            
        elif solver == 'heun':
            # Predictor Step (Euler)
            I_temp_raw = I_t + u_t * dt
            
            # Harmonization (For OT, S=0, so this just keeps I_temp_raw)
            I_temp = S * (y_obs + sig_next * epsilon) + (1 - S) * I_temp_raw
            
            # Corrector Network Evaluation
            with torch.no_grad():
                model_out_next, _ = model(x, I_temp, t_next_tensor, S)
            
            if pred_target == 'y':
                beta_next = beta_fn(t_next_tensor, schedule_type)
                dbeta_next = dbeta_dt_fn(t_next_tensor, schedule_type)
                dsig_next = dsigma_dt_fn(t_next_tensor, sigma_max, schedule_type)
                
                ratio_next = torch.where(sig_next > 1e-7, dsig_next / sig_next, torch.zeros_like(sig_next))
                y_coeff_next = S + (1 - S) * beta_next
                
                u_t_next = (1 - S) * dbeta_next * model_out_next + ratio_next * (I_temp - y_coeff_next * model_out_next)
            elif pred_target in ['v', 'ot']:
                u_t_next = model_out_next
                
            # Final Corrector Step
            I_raw = I_t + 0.5 * (u_t + u_t_next) * dt
        
        # STRICT HARMONIZATION: Re-inject exact theoretical trajectory
        # For OT (S=0), this cleanly collapses to I_t = I_raw
        I_t = S * (y_obs + sig_next * epsilon) + (1 - S) * I_raw
        
    return I_t

# =============================================================================
# IMPLICIT TIME-INDEPENDENT FLOW (CFM-0) - ASYMMETRIC FORMULATION
# =============================================================================

def inference_cfm_0(model, x, eta=0.1, max_steps=1000, eps_conv=1e-7, 
                    S=None, y_obs=None, pred_target='y'):
    """
    Executes autonomous fixed-point iteration for time-independent generation.
    
    Args:
        target_type: "y" for y-prediction (state), "v" for v-prediction (velocity).
    """
    batch_size = x.size(0)
    
    if S is None:
        S = torch.zeros(batch_size, model.y_dim, device=x.device)
        y_obs = torch.zeros(batch_size, model.y_dim, device=x.device)
        
    # ASYMMETRIC INITIALIZATION:
    # Observed (S=1) are clean ground truth; unobserved (S=0) are noise.
    epsilon = torch.randn(batch_size, model.y_dim, device=x.device)
    y_k = S * y_obs + (1 - S) * epsilon
    
    converged_mask = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
    
    for step in range(max_steps):
        if converged_mask.all():
            break
            
        with torch.no_grad():
            net_out, _ = model(x, y_k, S)
        
        # Derive transport vector based on target formulation
        if pred_target == "v":
            v = net_out
        elif pred_target == "y":
            v = net_out - y_k
        else:
            raise ValueError(f"Unknown target_type: {target_type}. Choose 'y' or 'v'.")
        
        norms = torch.norm(v, p=2, dim=1)
        converged_mask = converged_mask | (norms < eps_conv)
        
        active = ~converged_mask
        if active.any():
            y_k_next_raw = y_k[active] + eta * v[active]
            y_k[active] = S[active] * y_obs[active] + (1 - S[active]) * y_k_next_raw
                
    return y_k