import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

import src.dataset as ds
import src.inference as infer
from src.physics_constraints import ProjectionSolver, compute_physics_residuals

class ExperimentEvaluator:
    def __init__(self, config, device, seed):
        self.config = config
        self.device = device
        self.seed = seed
        self.sigma_max = float(config['model'].get('sigma_max', 1.0))
        
        torch.manual_seed(seed)
        np.random.seed(seed)

        data_dict = ds.prepare_data_model(config['data']["data_path"], config['training']["ratio_test_val_train"])
        self.X_test, self.y_test = data_dict['test']
        self.scaler_X, self.scaler_Y = data_dict['scalers']
        
        data_dict = ds.prepare_data_model(config['data']["data_path"], config['training']["ratio_test_val_train"])

        # Initialize the Hard Projection Solver from CasADi
        self.proj_solver = ProjectionSolver(self.scaler_X, self.scaler_Y, 
                                            x_dim=config['model']['x_dim'], p_dim=config['model']['y_dim'])
        
        test_dataset = ds.LTPDataset(self.X_test, self.y_test)
        self.test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    def _compute_errors(self, x_np, y_true_np, y_pred_np):
        """Returns Data RMSE and Physics RMSE"""
        data_rmse = np.sqrt(((y_pred_np - y_true_np)**2).mean())
        
        x_tensor = torch.from_numpy(x_np).to(device=self.device, dtype=torch.float32)
        y_pred_tensor = torch.from_numpy(y_pred_np).to(device=self.device, dtype=torch.float32)
        
        # Calculate Physics Loss using the exact vectorized tensor math
        phys_loss = compute_physics_residuals(x_tensor, y_pred_tensor, self.scaler_X, self.scaler_Y).pow(2).mean().item()
        phys_rmse = np.sqrt(phys_loss)
        
        return data_rmse, phys_rmse

    # Add observed_indices=None to the method signature
    def evaluate_generative_model(self, model, mode="CFM_t", observed_indices=None): 
        """Evaluates CFM_t or CFM_0 models for Surrogate or Imputation tasks"""
        all_y_true, all_x, all_raw_pred = [], [], []
        model.eval()
        
        for x, y in self.test_dataloader:
            x = x.to(self.device)
            y = y.to(self.device) # Move true y to device to extract observed values
            
            batch_size = x.shape[0]
            
            # 1. Initialize empty mask (Surrogate Model default: everything is missing)
            S_mask = torch.zeros(batch_size, self.proj_solver.p_dim, device=self.device)
            y_obs = torch.zeros(batch_size, self.proj_solver.p_dim, device=self.device)
            
            # 2. If specific indices are provided, update the mask for Imputation
            if observed_indices is not None and len(observed_indices) > 0:
                S_mask[:, observed_indices] = 1.0
                y_obs = y # The inference loop uses S * y_obs, so passing full y is perfectly safe
            
            if mode == "CFM_t": 
                N_steps = self.config['inference'].get('steps_t', 100)
                schedule_type = self.config['model'].get('schedule', 'linear')
                solver = self.config['inference'].get('solver', 'heun')
                pred_target = self.config['model'].get('pred_target', 'y')
                
                y_pred = infer.inference_cfm_t(
                    model, x, self.sigma_max, N_steps=N_steps, 
                    schedule_type=schedule_type, solver=solver,
                    S=S_mask, y_obs=y_obs, pred_target=pred_target   # Pass the dynamic mask and observed values
                )
                
            elif mode == "CFM_0": 
                max_steps = self.config['inference'].get('max_steps', 200)
                pred_target = self.config['model'].get('pred_target', 'y')
                
                y_pred = infer.inference_cfm_0(
                    model, x, max_steps=max_steps, S=S_mask, y_obs=y_obs, pred_target=pred_target
                )
                
            else:
                raise ValueError(f"Unknown mode in evaluate_generative_model: {mode}")
                
            all_x.append(x.cpu().numpy())
            all_y_true.append(y.cpu().numpy())
            all_raw_pred.append(y_pred.cpu().numpy())
            
        return self._process_predictions(all_x, all_y_true, all_raw_pred)


    def evaluate_regressor(self, regressor_model):
        """Evaluates the Baseline NN or NN+Soft models"""
        all_y_true, all_x, all_raw_pred = [], [], []
        regressor_model.eval()

        with torch.no_grad():
            for x, y in self.test_dataloader:
                x = x.to(self.device)
                y_pred = regressor_model(x)
                
                all_x.append(x.cpu().numpy())
                all_y_true.append(y.numpy())
                all_raw_pred.append(y_pred.cpu().numpy())

        return self._process_predictions(all_x, all_y_true, all_raw_pred)
    
    
    def _process_predictions(self, all_x, all_y_true, all_raw_pred, S_mask=None):
        """Helper to run metrics on raw outputs, apply projection, and run metrics again."""
        x_np = np.concatenate(all_x, axis=0)
        y_true_np = np.concatenate(all_y_true, axis=0)
        y_raw_np = np.concatenate(all_raw_pred, axis=0)
        
        # If no mask was provided (e.g. pure surrogate task or Regressor), create an all-zeros mask
        if S_mask is None:
            S_mask_np = np.zeros_like(y_raw_np)
        else:
            S_mask_np = S_mask.cpu().numpy()
        
        # Metrics BEFORE Projection
        data_err_raw, phys_err_raw = self._compute_errors(x_np, y_true_np, y_raw_np)
        
        # Hard Projection (Now passing the mask to lock observed variables!)
        y_proj_np = self.proj_solver.solve_batch(x_np, y_raw_np, S_mask_batch=S_mask_np)
        
        # Metrics AFTER Projection 
        data_err_proj, phys_err_proj = self._compute_errors(x_np, y_true_np, y_proj_np)
        
        return {
            'metrics': {
                'data_rmse_raw': data_err_raw, 
                'phys_rmse_raw': phys_err_raw,
                'data_rmse_proj': data_err_proj, 
                'phys_rmse_proj': phys_err_proj
            },
            'arrays': {
                'x': x_np,
                'y_true': y_true_np,
                'y_raw': y_raw_np,
                'y_proj': y_proj_np
            }
        }
    
    
    def evaluate_uq_dataset(self, model, observed_indices=None, num_samples=32, mode="CFM_t"):
        """
        Runs ensemble generation over the entire test dataset.
        Computes the relative residuals in the original physical scale.
        """
        model.eval()
        all_y_true, all_y_mean, all_y_std = [], [], []
        
        print(f"\n--- Running UQ Ensemble (Samples per point: {num_samples}) ---")
        
        with torch.no_grad():
            for x, y in self.test_dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                batch_size = x.size(0)
                
                # Expand the batch: [x1, x1... x2, x2... xN, xN...]
                x_expanded = x.repeat_interleave(num_samples, dim=0)
                y_expanded = y.repeat_interleave(num_samples, dim=0)
                
                # Construct mask
                S_mask = torch.zeros(batch_size * num_samples, self.proj_solver.p_dim, device=self.device)
                if observed_indices is not None and len(observed_indices) > 0:
                    S_mask[:, observed_indices] = 1.0
                
                # Run generative inference
                if mode == "CFM_t":
                    N_steps = self.config['inference'].get('steps_t', 100)
                    schedule_type = self.config['model'].get('schedule', 'linear')
                    solver = self.config['inference'].get('solver', 'heun')
                    pred_target = self.config['model'].get('pred_target', 'y')
                    
                    y_preds = infer.inference_cfm_t(
                        model, x_expanded, self.sigma_max, N_steps=N_steps, 
                        schedule_type=schedule_type, solver=solver,
                        S=S_mask, y_obs=y_expanded, pred_target=pred_target
                    )
                elif mode == "CFM_0":
                    max_steps = self.config['inference'].get('steps_0', 200)
                    schedule_type = self.config['model'].get('schedule', 'linear')
                    
                    y_preds = infer.inference_cfm_0(
                        model, x_expanded, max_steps=max_steps, S=S_mask, y_obs=y_expanded
                    )
                
                # Reshape back to (batch_size, num_samples, y_dim)
                y_preds = y_preds.view(batch_size, num_samples, -1).cpu().numpy()
                y_true = y.cpu().numpy()
                
                # Calculate mean and standard deviation over the ensemble dimension
                y_mean = np.mean(y_preds, axis=1)
                y_std = np.std(y_preds, axis=1)
                
                all_y_true.append(y_true)
                all_y_mean.append(y_mean)
                all_y_std.append(y_std)
                
        # Concatenate full dataset
        y_true_np = np.concatenate(all_y_true, axis=0)
        y_mean_np = np.concatenate(all_y_mean, axis=0)
        y_std_np = np.concatenate(all_y_std, axis=0)
        
        # Unscale to physical units to compute meaningful relative residuals
        y_true_phys = self.scaler_Y.inverse_transform(y_true_np)
        y_mean_phys = self.scaler_Y.inverse_transform(y_mean_np)
        y_std_phys = y_std_np * self.scaler_Y.scale_ # Scale the standard deviation
        
        # Calculate Relative Residuals: (Pred - True) / |True|
        # Add epsilon to avoid divide-by-zero on edge cases
        eps = 1e-8
        relative_residuals = (y_mean_phys - y_true_phys) / (np.abs(y_true_phys) + eps)
        
        return {
            'relative_residuals': relative_residuals,
            'y_std_phys': y_std_phys,
            'y_mean_phys': y_mean_phys,
            'y_true_phys': y_true_phys
        }