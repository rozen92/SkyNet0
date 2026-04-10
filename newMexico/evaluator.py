import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from models import BladeMLP

def plot_and_save(data_ctx, Y_all_pred, rmse_dict, hist, best_cv_loss, test_l_sc, title, filename, image_dir):
    """Fonction générique pour tracer les graphiques (MLP ou Interpolation)."""
    is_hybrid, is_global = data_ctx['is_hybrid'], data_ctx['is_global']
    forces_data = data_ctx['forces_data']
    azimuth_angles, r_centers_norm = data_ctx['azimuth_angles'], data_ctx['r_centers_norm']
    X_train_full, X_test = data_ctx['X_train_full'], data_ctx['X_test']
    Yf_train_full, Yf_test = data_ctx['Yf_train_full'], data_ctx['Yf_test']
    Y_train_full, Y_test = data_ctx['Y_train_full'], data_ctx['Y_test']

    Fn_data, Ft_data = forces_data[0], forces_data[1]
    Veff_data, Alpha_data = forces_data[2], forces_data[3]

    n_rows = 3 if is_hybrid else 2
    fig = plt.figure(figsize=(16, 5 * n_rows))
    plt.subplots_adjust(hspace=0.45)

    # 1. Graphe de Loss (Uniquement si historique fourni, sinon vide pour l'interpolation)
    ax_loss = plt.subplot(n_rows, 2, 1)
    if hist is not None:
        ax_loss.plot(hist, color='purple', label='Train Loss')
        ax_loss.axhline(y=best_cv_loss, color='orange', ls='--', label=f'Best CV MSE: {best_cv_loss:.4f}')
        ax_loss.axhline(y=test_l_sc, color='red', ls='-.', label=f'Test MSE: {test_l_sc:.4f}')
        ax_loss.set_yscale('log')
        ax_loss.set_title("Convergence")
        ax_loss.set_xlabel("Époques")
        ax_loss.set_ylabel("MSE (Scaled)")
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)
    else:
        ax_loss.set_title("Pas de convergence (Méthode Directe)")
        ax_loss.axis('off')

    # 2. Légende Globale Commune (En haut à droite)
    sections = np.linspace(0, 33, 7, dtype=int)
    method_proxies = [Line2D([0], [0], color='black', lw=2, label='Prédiction Modèle'),
                      Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', alpha=0.3, label='SVEN Entraînement'),
                      Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markeredgecolor='black', label='SVEN Test')]
    section_proxies = [Line2D([0], [0], color=plt.cm.viridis(i/len(sections)), lw=3, label=f'Section {s}') for i, s in enumerate(sections)]
    ax_leg = plt.subplot(n_rows, 2, 2)
    ax_leg.axis('off')
    ax_leg.legend(handles=method_proxies + section_proxies, loc='center', ncol=2, fontsize='medium', frameon=True, title="Légende Commune")

    # 3. Préparation des grilles de prédiction (remaniement des tenseurs)
    if is_hybrid:
        V_grid = Y_all_pred[:, 0].reshape(72, 34) if not is_global else Y_all_pred[:, :34]
        A_grid = np.degrees(Y_all_pred[:, 1].reshape(72, 34)) if not is_global else np.degrees(Y_all_pred[:, 34:])
        Fn_grid = forces_data[0]*(V_grid/forces_data[2])**2
        Ft_grid = forces_data[1]*(V_grid/forces_data[2])**2
    else:
        Fn_grid = Y_all_pred[:,0].reshape(72,34) if not is_global else Y_all_pred[:, :34]
        Ft_grid = Y_all_pred[:,1].reshape(72,34) if not is_global else Y_all_pred[:, 34:]

    axes_map = {'Fn': (n_rows, 2, 3), 'Ft': (n_rows, 2, 4), 'Veff': (n_rows, 2, 5), 'Alpha': (n_rows, 2, 6)}
    
    ylabels = {
        'Fn': 'Force $F_n$ [N/m]', 
        'Ft': 'Force $F_t$ [N/m]', 
        'Veff': 'Vitesse $V_{eff}$ [m/s]', 
        'Alpha': 'Angle $\\alpha$ [°]'
    }

    # Boucle sur chaque type de grandeur à afficher
    for key, (r, c, pos) in axes_map.items():
        if key in ['Veff', 'Alpha'] and not is_hybrid: continue
        
        ax = plt.subplot(r, c, pos)
        for i, s in enumerate(sections):
            col = plt.cm.viridis(i/len(sections))
            val_grid = {'Fn': Fn_grid, 'Ft': Ft_grid, 'Veff': V_grid if is_hybrid else None, 'Alpha': A_grid if is_hybrid else None}[key]
            
            # Tracé de la courbe continue du modèle
            ax.plot(azimuth_angles, val_grid[:, s], color=col, lw=1.5)

            # Identification des indices pour l'azimut (dépend du type de modèle)
            idx_in, idx_cos = (2, 1) if not is_global else (1, 0)
            tr_ang = (np.degrees(np.arctan2(X_train_full[:, idx_in], X_train_full[:, idx_cos])) % 360)
            te_ang = (np.degrees(np.arctan2(X_test[:, idx_in], X_test[:, idx_cos])) % 360)
            
            if not is_global:
                # Masques stricts pour isoler les points appartenant à la section 's'
                m_tr = np.isclose(X_train_full[:, 0], r_centers_norm[s], atol=1e-4)
                m_te = np.isclose(X_test[:, 0], r_centers_norm[s], atol=1e-4)
                
                tr_a, te_a = tr_ang[m_tr], te_ang[m_te]
                if key == 'Fn': tr_val, te_val = Yf_train_full[m_tr, 0], Yf_test[m_te, 0]
                elif key == 'Ft': tr_val, te_val = Yf_train_full[m_tr, 1], Yf_test[m_te, 1]
                elif key == 'Veff': tr_val, te_val = Y_train_full[m_tr, 0], Y_test[m_te, 0]
                elif key == 'Alpha': tr_val, te_val = np.degrees(Y_train_full[m_tr, 1]), np.degrees(Y_test[m_te, 1])
            else:
                tr_a, te_a = tr_ang, te_ang
                if key == 'Fn': tr_val, te_val = Yf_train_full[:, s], Yf_test[:, s]
                elif key == 'Ft': tr_val, te_val = Yf_train_full[:, s+34], Yf_test[:, s+34]
                elif key == 'Veff': tr_val, te_val = Y_train_full[:, s], Y_test[:, s]
                elif key == 'Alpha': tr_val, te_val = np.degrees(Y_train_full[:, s+34]), np.degrees(Y_test[:, s+34])

            # Tracé des points d'entraînement (ronds) et de test (étoiles)
            ax.scatter(tr_a, tr_val, marker='o', s=20, color=col, alpha=0.3)
            ax.scatter(te_a, te_val, marker='*', s=80, color='red', edgecolor='black', zorder=5)

        # Génération du titre avec les RMSE
        label_math = {'Fn': '$F_n$', 'Ft': '$F_t$', 'Veff': '$V_{eff}$', 'Alpha': '$\\alpha$'}[key]
        unit = {'Fn': 'N/m', 'Ft': 'N/m', 'Veff': 'm/s', 'Alpha': '°'}[key]
        
        t = f"{label_math}: RMSE {rmse_dict[key]['val']:.3f} {unit} ({rmse_dict[key]['err']:.1f}%)"
        
        ax.set_title(t)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("Azimut [°]")
        ax.set_ylabel(ylabels[key])

    plt.suptitle(title, fontsize=16, y=0.99)
    filepath = os.path.join(image_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" Image enregistrée : {filename}")

def evaluate_mlp(data_ctx, best_params, best_cv_loss, activation, model_type, image_dir, device):
    """Entraîne et évalue le MLP final avec les hyperparamètres optimaux."""
    final_model = BladeMLP(data_ctx['in_feats'], data_ctx['out_feats'], best_params['n_layers'], 
                           best_params['n_units'], best_params['dropout_rate'], activation).to(device)
    opt = optim.Adam(final_model.parameters(), lr=best_params['lr'])
    crit = nn.MSELoss()
    hist = []
    
    X_tr_t = torch.tensor(data_ctx['X_train_full'], dtype=torch.float32).to(device)
    Y_tr_t = torch.tensor(data_ctx['Y_train_full_scaled'], dtype=torch.float32).to(device)
    
    for _ in range(1000):
        final_model.train()
        opt.zero_grad()
        l = crit(final_model(X_tr_t), Y_tr_t)
        l.backward()
        opt.step()
        hist.append(l.item())

    final_model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(data_ctx['X_test'], dtype=torch.float32).to(device)
        test_l_sc = crit(final_model(X_test_t), torch.tensor(data_ctx['scaler_Y'].transform(data_ctx['Y_test']), dtype=torch.float32).to(device)).item()
        
        Y_p_test = data_ctx['scaler_Y'].inverse_transform(final_model(X_test_t).cpu().numpy())
        Y_all_pred = data_ctx['scaler_Y'].inverse_transform(final_model(torch.tensor(data_ctx['X'], dtype=torch.float32).to(device)).cpu().numpy())

    return compute_metrics_and_plot(data_ctx, Y_p_test, Y_all_pred, model_type, f"MLP {activation} | Modèle {model_type}", f"MLP_Model_{model_type}_{activation}.png", image_dir, hist, best_cv_loss, test_l_sc)


def evaluate_interpolation(data_ctx, model_type, image_dir):
    """Évalue la baseline d'interpolation classique de type Scipy."""
    print(f"\n--- Baseline Interpolation (Modèle {model_type}) ---")
    interp_lin = LinearNDInterpolator(data_ctx['X_train_full'], data_ctx['Y_train_full'])
    interp_near = NearestNDInterpolator(data_ctx['X_train_full'], data_ctx['Y_train_full'])

    # Prédiction Test
    Y_p_test = interp_lin(data_ctx['X_test'])
    mask = np.isnan(Y_p_test[:, 0])
    if np.any(mask): 
        Y_p_test[mask] = interp_near(data_ctx['X_test'][mask])

    # Prédiction All (pour les courbes)
    Y_all_pred = interp_lin(data_ctx['X'])
    mask_all = np.isnan(Y_all_pred[:, 0])
    if np.any(mask_all): 
        Y_all_pred[mask_all] = interp_near(data_ctx['X'][mask_all])

    return compute_metrics_and_plot(data_ctx, Y_p_test, Y_all_pred, model_type, f"Interpolation Baseline | Modèle {model_type}", f"Interp_Model_{model_type}.png", image_dir, None, 0, 0)

def compute_metrics_and_plot(data_ctx, Y_p_test, Y_all_pred, model_type, title_prefix, filename, image_dir, hist, best_cv_loss, test_l_sc):
    """Calcule les RMSE, les erreurs relatives et lance le tracé."""
    is_hybrid, is_global = data_ctx['is_hybrid'], data_ctx['is_global']
    Y_test, Yf_test = data_ctx['Y_test'], data_ctx['Yf_test']
    
    rmse_dict = {'Fn': {}, 'Ft': {}, 'Veff': {}, 'Alpha': {}}

    if is_hybrid:
        Vp, Vt = (Y_p_test[:, 0], Y_test[:, 0]) if not is_global else (Y_p_test[:, :34], Y_test[:, :34])
        Ap, At = (Y_p_test[:, 1], Y_test[:, 1]) if not is_global else (Y_p_test[:, 34:], Y_test[:, 34:])
        Fn_tr, Ft_tr = (Yf_test[:, 0], Yf_test[:, 1]) if not is_global else (Yf_test[:, :34], Yf_test[:, 34:])
        
        Fn_p_test = Fn_tr * (Vp / Vt)**2
        Ft_p_test = Ft_tr * (Vp / Vt)**2
        
        rmse_dict['Veff']['val'] = np.sqrt(np.mean((Vt - Vp)**2))
        rmse_dict['Veff']['err'] = (rmse_dict['Veff']['val'] / np.mean(np.abs(Vt)))*100
        
        rmse_dict['Alpha']['val'] = np.sqrt(np.mean((np.degrees(At - Ap))**2))
        rmse_dict['Alpha']['err'] = (rmse_dict['Alpha']['val'] / np.mean(np.abs(np.degrees(At))))*100
    else:
        Fn_p_test, Ft_p_test = (Y_p_test[:, 0], Y_p_test[:, 1]) if not is_global else (Y_p_test[:, :34], Y_p_test[:, 34:])
        Fn_tr, Ft_tr = (Yf_test[:, 0], Yf_test[:, 1]) if not is_global else (Yf_test[:, :34], Yf_test[:, 34:])

    rmse_dict['Fn']['val'] = np.sqrt(np.mean((Fn_tr - Fn_p_test)**2))
    rmse_dict['Fn']['err'] = (rmse_dict['Fn']['val'] / np.mean(np.abs(Fn_tr)))*100
    
    rmse_dict['Ft']['val'] = np.sqrt(np.mean((Ft_tr - Ft_p_test)**2))
    rmse_dict['Ft']['err'] = (rmse_dict['Ft']['val'] / np.mean(np.abs(Ft_tr)))*100
    
    r_glob = np.sqrt(0.5*(rmse_dict['Fn']['val']**2 + rmse_dict['Ft']['val']**2))

    full_title = f"{title_prefix} | RMSE Globale: {r_glob:.2f} N/m"
    plot_and_save(data_ctx, Y_all_pred, rmse_dict, hist, best_cv_loss, test_l_sc, full_title, filename, image_dir)