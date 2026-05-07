import torch
from torch.nn import functional as F

def compute_transfer_matrix_saliency(model, x, target_channel, baseline_type='identity', steps=50, delay=0, metric="mse"):
    """
    Computes Internal Integrated Gradients for the transfer matrices of DECIFRA.
    
    Args:
        model: DECIFRA or DECIFRA_MS instance
        x: Input tensor [B, T, C]
        target_channel: int, the channel whose forecast error we want to explain. If -1, explains error over all channels.
        baseline_type: 'identity' or 'zero'
        steps: Number of interpolation steps for IG
        delay: Prediction delay used in the loss calculation
        metric: 'mse', 'mae', or 'forecast'
        
    Returns:
        saliency_map: Tensor of shape [B, T, C, C] containing the Integrated Gradients
    """
    model.eval()
    
    B, T, C = x.shape
    
    # 1. Run standard forward pass to get actual transfer matrices
    with torch.no_grad():
        _, loss_load_true = model(x)
        matrices_true = loss_load_true["matrices"] # [B, T, C, C]
        
    # 2. Define baseline
    if baseline_type == 'identity':
        matrices_base = torch.eye(C, device=x.device).unsqueeze(0).unsqueeze(0).expand(B, T, C, C)
    elif baseline_type == 'zero':
        matrices_base = torch.zeros_like(matrices_true)
    else:
        raise ValueError(f"Unknown baseline_type: {baseline_type}")
        
    # 3. Compute gradients over interpolated path
    integrated_gradients = torch.zeros_like(matrices_true)
    
    # Target signal is fixed
    if target_channel == -1:
        target_signal = x[:, 1+delay:, :]
    else:
        target_signal = x[:, 1+delay:, target_channel]
    
    # Loop over alpha steps
    # According to standard IG, alpha goes from 1/steps to 1
    # We can use a summation approximation
    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=x.device)
    
    for alpha in alphas:
        # Interpolate matrices
        matrices_alpha = matrices_base + alpha * (matrices_true - matrices_base)
        matrices_alpha.requires_grad_(True)
        
        # Run modified forward pass
        _, loss_load_alpha = model(x, forced_matrices=matrices_alpha)
        
        if target_channel == -1:
            predicted_signal = loss_load_alpha["predicted"][:, delay:, :]
        else:
            predicted_signal = loss_load_alpha["predicted"][:, delay:, target_channel]
        
        # Compute error metric
        if metric == "mse":
            error = F.mse_loss(predicted_signal, target_signal, reduction='sum')
        elif metric == "mae":
            error = F.l1_loss(predicted_signal, target_signal, reduction='sum')
        elif metric == "forecast":
            error = predicted_signal.sum()
        else:
            raise ValueError(f"Unknown metric: {metric}")
            
        # Compute gradients
        grad = torch.autograd.grad(error, matrices_alpha)[0] # [B, T, C, C]
        
        integrated_gradients += grad
        
    # Average the gradients
    integrated_gradients /= steps
    
    # Multiply by (x - x')
    saliency_map = (matrices_true - matrices_base) * integrated_gradients
    
    return saliency_map
