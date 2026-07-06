import torch
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import norm

class PrivacyAttackSuite:
    def __init__(self, target_model, shadow_models, device):
        self.target = target_model
        self.shadows = shadow_models 
        self.device = device

    def shokri_shadow_mia(self, member_loader, non_member_loader):
        shadow_data, shadow_labels = [], []
        
        for model in self.shadows:
            model.eval()
            with torch.no_grad():
                for x, _ in member_loader:
                    x = x.to(self.device)
                    out = F.softmax(model(x), dim=1)
                    shadow_data.extend(out.cpu().numpy())
                    shadow_labels.extend([1] * x.size(0))
                for x, _ in non_member_loader:
                    x = x.to(self.device)
                    out = F.softmax(model(x), dim=1)
                    shadow_data.extend(out.cpu().numpy())
                    shadow_labels.extend([0] * x.size(0))

        X = np.array(shadow_data)
        y = np.array(shadow_labels)
        
        attacker_clf = RandomForestClassifier(n_estimators=100)
        attacker_clf.fit(X, y)
        
        # Evaluate RF on target model outputs for both members and non-members (Section IV-B)
        target_out, target_labels_list = [], []
        self.target.eval()
        with torch.no_grad():
            for x, _ in member_loader:
                x = x.to(self.device)
                out = F.softmax(self.target(x), dim=1)
                target_out.extend(out.cpu().numpy())
                target_labels_list.extend([1] * x.size(0))
            for x, _ in non_member_loader:
                x = x.to(self.device)
                out = F.softmax(self.target(x), dim=1)
                target_out.extend(out.cpu().numpy())
                target_labels_list.extend([0] * x.size(0))

        target_labels = np.array(target_labels_list)
        preds = attacker_clf.predict_proba(np.array(target_out))[:, 1]
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(target_labels, preds)

    def carlini_lira_attack(self, x_target, y_target):
        scores = []
        self.target.eval() 
        
        for model in self.shadows:
            model.eval()
            with torch.no_grad():
                out = F.softmax(model(x_target.to(self.device)), dim=1)
                conf = out[0, y_target].item()
                scores.append(np.log(conf / (1 - conf + 1e-10)))
        
        mu = np.mean(scores)
        std = np.std(scores) + 1e-10
        
        with torch.no_grad():
            t_out = F.softmax(self.target(x_target.to(self.device)), dim=1)
            t_conf = t_out[0, y_target].item()
            t_score = np.log(t_conf / (1 - t_conf + 1e-10))
            
        p_val = norm.cdf(t_score, mu, std)
        return p_val
