import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import src.models as models
import src.dataset as ds
import src.physics_constraints as constraints
import src.losses as losses 
from src.inference import sigma_fn

# =============================================================================
# HELPER: MODEL PARAMETER COUNT
# =============================================================================
def print_model_parameters(model, model_name="Model"):
    """Prints the total number of trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{model_name}] Total Trainable Parameters: {total_params:,}")
    return total_params

# =============================================================================
# 1. GENERATIVE MODEL TRAINING (CFM-t & CFM-0)
# =============================================================================
def train_cfm(seed, device, config, is_autonomous=False):
    """Unified training loop for CFM-t (time-dependent) and CFM-0 (autonomous)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    data_dict = ds.prepare_data_model(config['data']["data_path"], config['training']["ratio_test_val_train"])
    X_scaled, y_scaled = data_dict['train']
    
    # DATA SCARCITY SLICING
    train_frac = float(config['training'].get('train_fraction', 1.0))
    if train_frac < 1.0:
        num_samples = int(len(X_scaled) * train_frac)
        
        indices = np.random.permutation(len(X_scaled))
        selected_indices = indices[:num_samples]
        
        # Apply the randomized indices to extract a representative subset
        X_scaled = X_scaled[selected_indices]
        y_scaled = y_scaled[selected_indices]
        
        print(f"Data Scarcity Mode: Training on a random subset of {num_samples} samples ({train_frac*100:.1f}%)")
    
    X_val_scaled, y_val_scaled = data_dict['val']
    
    batch_size = config['training']['batch_size']
    dataloader = DataLoader(ds.LTPDataset(X_scaled, y_scaled), batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(ds.LTPDataset(X_val_scaled, y_val_scaled), batch_size=batch_size, shuffle=False)
    
    x_dim, y_dim = config['model']['x_dim'], config['model']['y_dim']
    hidden_dim = config['model']['hidden_dim']
    sigma_max = float(config['model']['sigma_max'])
    schedule_type = config['model'].get('schedule', 'linear')
    s_noise = config['model'].get('s_noise', 0.008)
    val_interval = config['training'].get('val_interval', 50)
    
    use_masking = config['training'].get('use_masking', True)
    pred_target = config['model'].get('pred_target', 'y') 
    
    if is_autonomous:
        model = models.CFM_0(x_dim=x_dim, y_dim=y_dim, hidden_dim=hidden_dim).to(device)
    else:
        model = models.CFM_t(x_dim=x_dim, y_dim=y_dim, hidden_dim=hidden_dim).to(device)
    
    print_model_parameters(model, 'CFM')    
    
    optimizer = optim.AdamW(model.parameters(), lr=float(config['training']['lr']), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['n_epochs_max'])
    ema = models.EMA(model, decay=0.999)
    stopper = models.EarlyStopping(patience=4, min_delta=1e-4, min_epochs=config['training']['n_epochs_min'])

    for epoch in range(1, config['training']['n_epochs_max'] + 1):
        model.train()
        train_loss = 0.0
        for x, y in dataloader:
            optimizer.zero_grad(set_to_none=True)
            x, y = x.to(device), y.to(device)
            t = torch.rand(x.size(0), 1, device=device)
            
            # Fetch analytical trajectories and exact targets
            I_t, target, S = losses.get_cfm_targets(
                y, t, sigma_max, schedule_type,
                use_masking, pred_target, is_autonomous
            )
            
            if is_autonomous:
                pred, _ = model(x, I_t, S)
            else:
                pred, _ = model(x, I_t, t, S)
                
            loss = nn.MSELoss()(pred, target)
            loss.backward()
            
            optimizer.step()
            ema.step()
            train_loss += loss.item()
            
        train_loss /= len(dataloader)
        scheduler.step()
            
        if epoch == 1 or epoch % val_interval == 0 or epoch == config['training']['n_epochs_max']:
            model.eval()
            val_loss = 0.0
            ema.apply_shadow()
            with torch.no_grad():
                for xv, yv in val_dataloader:
                    xv, yv = xv.to(device), yv.to(device)
                    t = torch.rand(xv.size(0), 1, device=device)
                    
                    I_t, target, S = losses.get_cfm_targets(
                        yv, t, sigma_max, schedule_type,
                        use_masking, pred_target, is_autonomous
                    )
                    
                    if is_autonomous:
                        pred, _ = model(xv, I_t, S)
                    else:
                        pred, _ = model(xv, I_t, t, S)
                        
                    val_loss += nn.MSELoss()(pred, target).item()
                    
            val_loss /= len(val_dataloader)
            ema.restore()
            print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.6f} | Val Loss (EMA): {val_loss:.6f}")
            if stopper.step(val_loss, epoch):
                break
                
    ema.apply_shadow()
    return model


# =============================================================================
# 2. BASELINE REGRESSOR TRAINING (NN & NN+Soft)
# =============================================================================
def train_regressor(seed, device, config):
    """Trains Standard Regressor (weight_physics=0) and Soft-Constrained Regressor (weight_physics>0)"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    data_dict = ds.prepare_data_model(config['data']["data_path"], config['training']["ratio_test_val_train"])
    X_scaled, y_scaled = data_dict['train']
    
    # DATA SCARCITY SLICING
    train_frac = float(config['training'].get('train_fraction', 1.0))
    if train_frac < 1.0:
        num_samples = int(len(X_scaled) * train_frac)
        
        indices = np.random.permutation(len(X_scaled))
        selected_indices = indices[:num_samples]
        
        # Apply the randomized indices to extract a representative subset
        X_scaled = X_scaled[selected_indices]
        y_scaled = y_scaled[selected_indices]
        
        print(f"Data Scarcity Mode: Training on a random subset of {num_samples} samples ({train_frac*100:.1f}%)")
    
    X_val_scaled, y_val_scaled = data_dict['val']  
    scaler_X, scaler_Y = data_dict['scalers']
    
    batch_size = config['training']['batch_size']
    dataloader = DataLoader(ds.LTPDataset(X_scaled, y_scaled), batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(ds.LTPDataset(X_val_scaled, y_val_scaled), batch_size=batch_size, shuffle=False, num_workers=0)
    
    val_interval = config['training'].get('val_interval', 50)
    
    regressor = models.Regressor(
        input_size=config['model']['x_dim'], 
        output_size=config['model']['y_dim'], 
        hidden_sizes=config['model']['hidden_size']
    ).to(device)
    
    weight_physics = float(config['training'].get('weights_physics', 0.0))
    model_name = "NN+Soft" if weight_physics > 0 else "NN"
    
    print_model_parameters(regressor, model_name)
    
    optimizer = optim.Adam(regressor.parameters(), lr=float(config['training']['lr']))
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=False)
    stopper = models.EarlyStopping(patience=20, min_delta=1e-4, min_epochs=config['training']['n_epochs_min'])

    for epoch in range(1, config['training']['n_epochs_max'] + 1):
        regressor.train()
        train_loss = 0.0
        for x, y in dataloader:
            optimizer.zero_grad(set_to_none=True)
            x, y = x.to(device), y.to(device)
            
            pred = regressor(x)
            loss_data = nn.MSELoss()(pred, y)
            
            if weight_physics > 0:
                loss_physics = constraints.compute_physics_residuals(x, pred, scaler_X, scaler_Y).pow(2).mean()
            else:
                loss_physics = 0.0
                
            total_loss = (1.0 - weight_physics) * loss_data + weight_physics * loss_physics
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            
        train_loss /= len(dataloader)
            
        # ---------------------------------------------------------
        # VALIDATION BLOCK (Now runs every epoch for the scheduler)
        # ---------------------------------------------------------
        regressor.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xv, yv in val_dataloader:
                xv, yv = xv.to(device), yv.to(device)
                pred = regressor(xv)
                
                loss_data = nn.MSELoss()(pred, yv)
                loss_physics = constraints.compute_physics_residuals(xv, pred, scaler_X, scaler_Y).pow(2).mean() if weight_physics > 0 else 0.0
                
                val_loss += ((1.0 - weight_physics) * loss_data + weight_physics * loss_physics).item()
                
        val_loss /= len(val_dataloader)
        
        # Step the scheduler based on the continuous validation loss
        scheduler.step(val_loss)
        
        # Print periodically (includes current Learning Rate to monitor the drops)
        if epoch == 1 or epoch % val_interval == 0 or epoch == config['training']['n_epochs_max']:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}")
            
        # Early Stopping Check
        if stopper.step(val_loss, epoch):
            print(f"[{model_name}] Early stopping triggered at epoch {epoch}. Final Val Loss: {val_loss:.6f}")
            break
            
    return regressor
