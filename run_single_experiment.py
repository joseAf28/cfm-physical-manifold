import yaml
import torch
import argparse
import numpy as np

import src.train as train
from src.experiment_evaluator import ExperimentEvaluator

def run_experiment(config_path, seed=42):
    # 1. Load Configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    model_type = config['model_name']
    print(f"Using device: {device}")
    print(f"--- Starting Individual Test for {model_type} (Seed: {seed}) ---")

    # 2. Train the Model
    if model_type == 'CFM_0':
        model = train.train_cfm(seed, device, config, is_autonomous=True)
    elif model_type == 'CFM_t':
        model = train.train_cfm(seed, device, config, is_autonomous=False)
    elif model_type == 'NN_Soft':
        model = train.train_regressor(seed, device, config)
    elif model_type == 'NN':
        model = train.train_regressor(seed, device, config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 3. Evaluate and Project
    print("\n--- Training Complete. Starting Inference and Hard Projection ---")
    evaluator = ExperimentEvaluator(config, device, seed)
    
    # Evaluate Standard Surrogate Task for all models
    if model_type in ['CFM_0', 'CFM_t']:
        results_surrogate = evaluator.evaluate_generative_model(model, mode=model_type)
    else:
        results_surrogate = evaluator.evaluate_regressor(model)
        
    print("\n--- Surrogate: Final Results ---")
    # UPDATED: Access the 'metrics' key
    metrics_surr = results_surrogate['metrics']
    print(f"Raw Generative Output -> Data RMSE: {metrics_surr['data_rmse_raw']:.6f} | Physics RMSE: {metrics_surr['phys_rmse_raw']:.2e}")
    print(f"After Hard Projection -> Data RMSE: {metrics_surr['data_rmse_proj']:.6f} | Physics RMSE: {metrics_surr['phys_rmse_proj']:.2e}")

    # Evaluate Imputation Tasks (Only available for Generative Models)
    if model_type in ['CFM_0', 'CFM_t']:
        results_imp1 = evaluator.evaluate_generative_model(model, mode=model_type, observed_indices=[1, 6])
        results_imp2 = evaluator.evaluate_generative_model(model, mode=model_type, observed_indices=[1, 2, 3, 4, 5, 6, 7])
        
        print("\n--- Imputation (k=2): Final Results ---")
        metrics_imp1 = results_imp1['metrics']
        print(f"Raw Generative Output -> Data RMSE: {metrics_imp1['data_rmse_raw']:.6f} | Physics RMSE: {metrics_imp1['phys_rmse_raw']:.2e}")
        print(f"After Hard Projection -> Data RMSE: {metrics_imp1['data_rmse_proj']:.6f} | Physics RMSE: {metrics_imp1['phys_rmse_proj']:.2e}")

        print("\n--- Imputation (k=7): Final Results ---")
        metrics_imp2 = results_imp2['metrics']
        print(f"Raw Generative Output -> Data RMSE: {metrics_imp2['data_rmse_raw']:.6f} | Physics RMSE: {metrics_imp2['phys_rmse_raw']:.2e}")
        print(f"After Hard Projection -> Data RMSE: {metrics_imp2['data_rmse_proj']:.6f} | Physics RMSE: {metrics_imp2['phys_rmse_proj']:.2e}")

        # ====================================================================
        # 4. DATASET-WIDE UNCERTAINTY & RELATIVE RESIDUALS
        # ====================================================================
        observed_cols = [1, 6] 
        
        uq_results = evaluator.evaluate_uq_dataset(
            model, observed_indices=observed_cols, num_samples=32, mode=model_type
        )
        
        rel_res = uq_results['relative_residuals']
        y_std = uq_results['y_std_phys']
        
        mean_rel_res = np.mean(rel_res, axis=0)
        std_rel_res = np.std(rel_res, axis=0)
        avg_uncertainty = np.mean(y_std, axis=0)
        
        print(f"\n--- Imputation UQ Statistics (Observed: {observed_cols}) ---")
        print(f"{'Var':<5} | {'Mean Rel. Error':<18} | {'Std Rel. Error':<18} | {'Mean UQ (Std Dev)':<18}")
        print("-" * 65)
        for i in range(rel_res.shape[1]):
            obs_marker = "*" if i in observed_cols else ""
            print(f"{str(i)+obs_marker:<5} | {mean_rel_res[i]:<18.4e} | {std_rel_res[i]:<18.4e} | {avg_uncertainty[i]:<18.4e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run specific model experiment via YAML config.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    run_experiment(args.config, args.seed)