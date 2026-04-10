import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
from models import BladeMLP

def find_best_hyperparameters(data_ctx, activation, n_trials, device):
    """Effectue une recherche Optuna avec K-Fold Cross Validation."""
    print(f"\n--- Recherche des meilleurs hyperparamètres ({activation})... ---")
    
    X_train_full = data_ctx['X_train_full']
    Y_train_full_scaled = data_ctx['Y_train_full_scaled']
    in_feats, out_feats = data_ctx['in_feats'], data_ctx['out_feats']

    def objective(trial):
        n_layers = trial.suggest_int('n_layers', 2, 8)
        n_units = trial.suggest_int('n_units', 64, 512)
        dr = trial.suggest_float('dropout_rate', 0.0, 0.2)
        lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
        
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        cv_l = []
        for tr_i, va_i in kf.split(X_train_full):
            m = BladeMLP(in_feats, out_feats, n_layers, n_units, dr, activation).to(device)
            opt = optim.Adam(m.parameters(), lr=lr)
            crit = nn.MSELoss()
            for _ in range(150):
                m.train(); opt.zero_grad()
                pred = m(torch.tensor(X_train_full[tr_i], dtype=torch.float32).to(device))
                targ = torch.tensor(Y_train_full_scaled[tr_i], dtype=torch.float32).to(device)
                crit(pred, targ).backward()
                opt.step()
            m.eval()
            with torch.no_grad():
                val_pred = m(torch.tensor(X_train_full[va_i], dtype=torch.float32).to(device))
                val_targ = torch.tensor(Y_train_full_scaled[va_i], dtype=torch.float32).to(device)
                cv_l.append(crit(val_pred, val_targ).item())
        return np.mean(cv_l)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params, study.best_value