import os
import yaml
import torch
import argparse
import src.train as train
from src.experiment_evaluator import ExperimentEvaluator

def train_and_save(config_path, output_dir, seed=42):
    # 1. Load Configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_type = config['model_name']
    
    print(f"--- Training {model_type} from config: {config_path} ---")

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

    # 3. Save the Checkpoint and the Config used
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract the unique experiment name from the config path 
    base_filename = os.path.splitext(os.path.basename(config_path))[0]
    
    # Now save them using the unique base filename
    model_path = os.path.join(output_dir, f"{base_filename}.pt")
    torch.save(model.state_dict(), model_path)
    
    config_out_path = os.path.join(output_dir, f"{base_filename}_config.yaml")
    with open(config_out_path, 'w') as f:
        yaml.dump(config, f)
        
    print(f"\nSuccess! Model and config saved to {output_dir} as {base_filename}")

    # ==========================================================
    # 4. QUICK DEBUG EVALUATION 
    # ==========================================================
    print(f"\n--- Quick Debug Evaluation for {model_type} (Seed: {seed}) ---")
    evaluator = ExperimentEvaluator(config, device, seed)
    
    if model_type in ['CFM_0', 'CFM_t']:
        results = evaluator.evaluate_generative_model(model, mode=model_type)
    else:
        results = evaluator.evaluate_regressor(model)
    
    metrics = results['metrics']
    
    print(f"RAW Output  -> Data RMSE: {metrics['data_rmse_raw']:.6f} | Physics RMSE: {metrics['phys_rmse_raw']:.2e}")
    print(f"PROJ Output -> Data RMSE: {metrics['data_rmse_proj']:.6f} | Physics RMSE: {metrics['phys_rmse_proj']:.2e}")
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and save a model checkpoint.')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    train_and_save(args.config, args.out_dir, args.seed)