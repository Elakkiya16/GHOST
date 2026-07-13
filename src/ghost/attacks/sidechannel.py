import time
import torch
import numpy as np

def evaluate_side_channel(model, input_batch, device, config):
    """
    Simulation of Side Channel Leakage.
    Measures Correlation between Input Features and Intermediate Activations.
    """
    model.eval()
    input_batch = input_batch.to(device)
    
    # 1. Time Analysis
    timings = []
    side_channel_samples = config.get('side_channel_samples', 5)
    for _ in range(side_channel_samples):
        if device.type == 'mps': torch.mps.synchronize()
        if device.type == 'cuda': torch.cuda.synchronize()
        
        start = time.perf_counter()
        _ = model(input_batch)
        
        if device.type == 'mps': torch.mps.synchronize()
        if device.type == 'cuda': torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    
    timing_variance = np.var(timings)

    # 2. CPA (Correlation Power Analysis Simulation)
    # Inside evaluate_side_channel in attacks_sidechannel.py
    with torch.no_grad():
        if hasattr(model, 'edge'):
            # Path for HybridCNN (GHOST)
            x = model.unshuffle(input_batch)
            activations = model.edge(x)
        elif hasattr(model, 'model'):
            # Path for BaselineCNN (The Sequential model)
            # Extract activations after the first conv/pool block
            m = model.model
            # Run the first 4 layers of the Sequential Baseline
            activations = m[:4](input_batch)
        else:
            activations = next(model.children())(input_batch)
            
    input_energy = input_batch.view(input_batch.size(0), -1).abs().sum(dim=1).cpu().numpy()
    act_energy = activations.view(activations.size(0), -1).abs().sum(dim=1).cpu().numpy()
    
    if np.std(input_energy) == 0 or np.std(act_energy) == 0:
        correlation = 0.0
    else:
        correlation = np.corrcoef(input_energy, act_energy)[0, 1]
    
    return timing_variance, abs(correlation)
