import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as ssim

def run_idlg(model, target_img, target_label, device, config):
    model.eval()
    target_img = target_img.to(device)
    target_label = target_label.to(device)

    out = model(target_img)
    loss = nn.CrossEntropyLoss()(out, target_label)

    real_grad = torch.autograd.grad(loss, model.parameters(), allow_unused=True)

    real_grad = [
        g.detach() if g is not None else torch.zeros_like(p)
        for g, p in zip(real_grad, model.parameters())
    ]

    # iDLG label extraction
    last_layer_grad = real_grad[-1]
    extracted_label = torch.argmin(last_layer_grad).item()

    # Optimization-based reconstruction
    dummy_data = torch.randn(target_img.size()).to(device).requires_grad_(True)
    optimizer = torch.optim.LBFGS([dummy_data], lr=0.1)

    for _ in range(config['dlg_iterations']):
        def closure():
            optimizer.zero_grad()
            dummy_out = model(dummy_data)
            dummy_loss = nn.CrossEntropyLoss()(dummy_out, torch.tensor([extracted_label]).to(device))
            dummy_grad = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True, allow_unused=True)
            dummy_grad = [
                g if g is not None else torch.zeros_like(p)
                for g, p in zip(dummy_grad, model.parameters())
            ]
            dist = sum(((dg - rg) ** 2).sum() for dg, rg in zip(dummy_grad, real_grad))
            dist.backward()
            return dist
        optimizer.step(closure)

    # Normalised MSE to [0,1] (Table III: GHOST 0.0021 vs baseline 0.0183)
    pixel_range = (target_img.max() - target_img.min()).item()
    pixel_range = max(pixel_range, 1e-8)
    raw_mse = torch.mean((target_img - dummy_data.detach()) ** 2).item()
    norm_mse = raw_mse / (pixel_range ** 2)

    # SSIM (Table III: GHOST 0.0894 vs baseline 0.3628)
    t_np = target_img.squeeze(0).detach().cpu().numpy()
    d_np = dummy_data.squeeze(0).detach().cpu().numpy()
    t_np = np.transpose(t_np, (1, 2, 0))  # C,H,W → H,W,C
    d_np = np.transpose(d_np, (1, 2, 0))
    ssim_val = ssim(t_np, d_np, data_range=float(t_np.max() - t_np.min()), channel_axis=2)

    return norm_mse, ssim_val, extracted_label == target_label.item()
