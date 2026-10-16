import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from skimage.metrics import structural_similarity as ssim

def run_mia_auc(model, member_loader, non_member_loader, device):
    model.eval()
    def get_conf(loader):
        confs = []
        with torch.no_grad():
            for x, _ in loader:
                out = model(x.to(device))
                prob = F.softmax(out, dim=1) if out.max() > 1.0 else out
                confs.append(prob.cpu().numpy())
        return np.vstack(confs)

    m_p = get_conf(member_loader)
    nm_p = get_conf(non_member_loader)
    X = np.vstack([m_p, nm_p])
    y = np.array([1]*len(m_p) + [0]*len(nm_p))
    
    preds = X.max(axis=1)
    return roc_auc_score(y, preds)

def run_dlg_ssim(model, image, device):
    """Deep Leakage from Gradients: SSIM 1.0 = Total Leakage, 0.0 = Secure"""
    model.eval()
    image = image.to(device)
    label = torch.LongTensor([1]).to(device) # Dummy label
    
    out = model(image)
    loss = nn.CrossEntropyLoss()(out, label)
    external_grad = torch.autograd.grad(loss, model.parameters(), allow_unused=True)
    external_grad = [g.detach() for g in external_grad if g is not None]

    dummy_data = torch.randn(image.size()).to(device).requires_grad_(True)
    optimizer = torch.optim.LBFGS([dummy_data], lr=0.1)
    
    for _ in range(10): # Iterations for DLG
        def closure():
            optimizer.zero_grad()
            dummy_out = model(dummy_data)
            dummy_loss = nn.CrossEntropyLoss()(dummy_out, label)
            dummy_grad = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True, allow_unused=True)
            dummy_grad = [g for g in dummy_grad if g is not None]
            
            grad_diff = sum(((dg - eg)**2).sum() for dg, eg in zip(dummy_grad, external_grad))
            grad_diff.backward()
            return grad_diff
        optimizer.step(closure)

    # Calculate SSIM between real and reconstructed
    real_np = image.detach().cpu().squeeze().numpy().mean(0)
    recon_np = dummy_data.detach().cpu().squeeze().numpy().mean(0)
    return ssim(real_np, recon_np, data_range=real_np.max() - real_np.min())
