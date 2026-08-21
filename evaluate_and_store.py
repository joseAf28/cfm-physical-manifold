import os
import yaml
import torch
import h5py
import argparse
import numpy as np

from src.models import build_model
from src.experiment_evaluator import ExperimentEvaluator

def evaluate_and_store(config_path, model_path, output_h5_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    model_type = config['model_name']
    seed = config.get('seed', 42)
    
    evaluator = ExperimentEvaluator(config, device, seed)
    
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)
    with h5py.File(output_h5_path, 'w') as h5f:
        y_dim = config['model']['y_dim']
        
        print(f"--- Starting Evaluation for {model_type} (Seed {seed}) ---")
        print(f"Saving to: {output_h5_path}\n")
        
        # --- MODIFIED: Optimize the 'k' loop ---
        # Baselines act purely as zero-shot surrogates (k=0). 
        # CFM models do both surrogate (k=0) and data imputation (k>0).
        k_range = range(y_dim) if model_type in ['CFM_t', 'CFM_0'] else [0]
        
        for k in k_range:
            print(f"Evaluating k={k} observed variables...")
            observed_indices = list(range(k)) if k > 0 else []
            
            if model_type in ['CFM_t', 'CFM_0']:
                results = evaluator.evaluate_generative_model(
                    model=model, 
                    mode=model_type, 
                    observed_indices=observed_indices
                )
            else: 
                results = evaluator.evaluate_regressor(model)
                
            arrays = results['arrays']
            metrics = results['metrics']
            
            print(f"  -> RAW  | Data RMSE: {metrics['data_rmse_raw']:.6f} | Physics RMSE: {metrics['phys_rmse_raw']:.2e}")
            print(f"  -> PROJ | Data RMSE: {metrics['data_rmse_proj']:.6f} | Physics RMSE: {metrics['phys_rmse_proj']:.2e}")
            print("-" * 60)
            
            if k == 0:
                h5f.create_dataset('x_cond', data=arrays['x'])
                h5f.create_dataset('y_true', data=arrays['y_true'])
            
            grp = h5f.create_group(f'k_observed_{k}')
            grp.create_dataset('y_raw', data=arrays['y_raw'])
            grp.create_dataset('y_proj', data=arrays['y_proj'])
            
            for metric_name, value in metrics.items():
                grp.attrs[metric_name] = value

    print(f"\nEvaluation completely finished and stored successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model_pt', type=str, required=True)
    parser.add_argument('--out_h5', type=str, required=True)
    args = parser.parse_args()
    
    evaluate_and_store(args.config, args.model_pt, args.out_h5)