import torch
from src.inference import alpha_fn, beta_fn, sigma_fn, dbeta_dt_fn, dsigma_dt_fn

def get_cfm_targets(y, t, sigma_max, schedule_type='quad', use_masking=True, pred_target='y', is_autonomous=False):
    """
    Constructs the interpolant (I_t) and the exact training target.
    Agnostically handles 'linear' vs 'quad' schedules for both CFM-t and CFM-0.
    """
    batch_size = y.size(0)
    device = y.device
    
    # ========================================================================
    # ENFORCE OT FLOW MATCHING BASELINE RULES
    # ========================================================================
    if pred_target == 'ot':
        use_masking = False
        schedule_type = 'linear'  # OT strictly requires a linear path
        sigma_max = 1.0
        
    # 1. Sample Mask S
    if use_masking:
        p = torch.rand(batch_size, 1, device=device)
        S = (torch.rand(batch_size, y.size(1), device=device) < p).float()
    else:
        S = torch.zeros_like(y)
        
    epsilon = torch.randn_like(y)
    
    # 2. Evaluate generalized schedules
    beta_t = beta_fn(t, schedule_type)
    sig_t = sigma_fn(t, sigma_max, schedule_type)
    dbeta_t = dbeta_dt_fn(t, schedule_type)
    dsig_t = dsigma_dt_fn(t, sigma_max, schedule_type)
    
    if pred_target == 'ot':
        # ====================================================================
        # STANDARD OPTIMAL TRANSPORT (OT) FLOW MATCHING BASELINE
        # ====================================================================
        # Maps pure independent noise directly to the target. No masking.
        I_t = beta_t * y + sig_t * epsilon
        
        # Target is the constant optimal transport velocity: y - x_0
        target = y - sigma_max * epsilon

    elif is_autonomous:
        # ====================================================================
        # TIME-INDEPENDENT (CFM-0): ASYMMETRIC INTERPOLANT
        # ====================================================================
        # Observed (S=1) remain fixed at clean y. Unobserved (S=0) move from noise.
        I_t = S * y + (1 - S) * (beta_t * y + sig_t * epsilon)
        
        if pred_target == 'y':
            target = y
        elif pred_target == 'v':
            target = (1 - S) * (dbeta_t * y + dsig_t * epsilon)
        else:
            raise ValueError(f"Unknown pred_target: {pred_target}")
            
    else:
        # ====================================================================
        # TIME-DEPENDENT (CFM-t): SYMMETRIC INTERPOLANT
        # ====================================================================
        # Both observed and unobserved variables follow the noise schedule.
        I_t = S * (y + sig_t * epsilon) + (1 - S) * (beta_t * y + sig_t * epsilon)
        
        if pred_target == 'y':
            target = y
        elif pred_target == 'v':
            target = S * (dsig_t * epsilon) + (1 - S) * (dbeta_t * y + dsig_t * epsilon)
        else:
            raise ValueError(f"Unknown pred_target: {pred_target}")

    return I_t, target, S