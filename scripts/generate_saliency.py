import os
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse

from src.models.DECIFRA_MS import DECIFRA_MS
from src.models.saliency import compute_transfer_matrix_saliency
from src.datasets.ukb_exp import load_data_exp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_channels", type=int, nargs='+', help="List of target channels to compute saliency for")
    args = parser.parse_args()

    # Paths
    log_dir = "assets/logs/1_pretrain-ukb_plus_1000-DECIFRA_MS-default/00"
    out_dir = "assets/saliency"
    os.makedirs(out_dir, exist_ok=True)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load Model
    print("Loading model...")
    with open(f"{log_dir}/best_epoch.txt", "r") as f:
        best_epoch = int(f.read().strip())
    
    model_cfg = OmegaConf.load(f"{log_dir}/model_config.yaml")
    model = DECIFRA_MS(model_cfg)
    
    # We use strict=False because some configs might have changed or we are missing some state. Usually it's fine.
    model.load_state_dict(torch.load(f"{log_dir}/checkpoints/model_{best_epoch}.pt", map_location='cpu'))
    model.to(device)
    model.eval()
    
    # Load Data
    print("Loading UKB subset 10000:11000...")
    data_dict, _ = load_data_exp()
    data = data_dict["data"][10000:11000] # shape [1000, 490, 53]
    
    x = torch.tensor(data, dtype=torch.float32)
    B, T, C = x.shape
    
    # Saliency parameters
    batch_size = 16
    
    if args.target_channels is not None:
        channels = args.target_channels
    else:
        channels = list(range(C)) + [-1]
    
    for c in channels:
        print(f"Computing saliency for target channel {c}...")
        all_saliency = []
        all_ig = []
        all_matrices_true = []
        
        for i in tqdm(range(0, B, batch_size)):
            x_batch = x[i:i+batch_size].to(device)
            
            s_map, i_grad, mat_true = compute_transfer_matrix_saliency(
                model, x_batch, target_channel=c, baseline_type='identity', steps=50, delay=0, metric='mse'
            )
            
            all_saliency.append(s_map.cpu().numpy())
            all_ig.append(i_grad.cpu().numpy())
            all_matrices_true.append(mat_true.cpu().numpy())
            
        all_saliency = np.concatenate(all_saliency, axis=0)
        all_ig = np.concatenate(all_ig, axis=0)
        all_matrices_true = np.concatenate(all_matrices_true, axis=0)
        
        # Save arrays
        sm_path = os.path.join(out_dir, f"saliency_map_channel_{c}.npy")
        ig_path = os.path.join(out_dir, f"integrated_grads_channel_{c}.npy")
        mat_path = os.path.join(out_dir, f"matrices_true_channel_{c}.npy")
        np.save(sm_path, all_saliency)
        np.save(ig_path, all_ig)
        np.save(mat_path, all_matrices_true)
        print(f"Saved to {sm_path}, {ig_path}, and {mat_path}")

if __name__ == "__main__":
    main()
