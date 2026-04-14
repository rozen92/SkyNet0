import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import pandas as pd

def load_base_data():
    """Charge les données à partir du fichier Excel pour un TSR et yaw spécifiques."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    dataset_path = os.path.join(current_dir, 'dataset_forces_mexico.xlsx')

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Fichier introuvable : {dataset_path}")

    # Lecture du fichier Excel
    print("Chargement du fichier Excel...")
    df = pd.read_excel(dataset_path)

    # 1. Filtrage du point de fonctionnement souhaité
    yaw_target = 15.0
    tsr_target = 8.0
    df_filtered = df[(df['yaw'] == yaw_target) & (df['TSR'] == tsr_target)].copy()

    if df_filtered.empty:
        raise ValueError(f"Aucune donnée trouvée pour yaw={yaw_target} et TSR={tsr_target}")

    # 2. Extraction des vecteurs uniques
    unique_r = np.sort(df_filtered['r'].unique())
    azimuth_angles = np.sort(df_filtered['theta'].unique())
    theta_rad = np.radians(azimuth_angles)

    # Normalisation du rayon r 
    r_centers_norm = unique_r / unique_r.max()

    # 3. Création des matrices 2D avec .pivot()
    # On pivote le tableau pour avoir theta en lignes et r en colonnes
    Fn_data = df_filtered.pivot(index='theta', columns='r', values='Fn').values
    Ft_data = df_filtered.pivot(index='theta', columns='r', values='Ft').values
    Veff_data = df_filtered.pivot(index='theta', columns='r', values='V_eff').values
    Alpha_data = df_filtered.pivot(index='theta', columns='r', values='alpha').values

    # 4. la matrice forces_data

    forces_data = np.stack([Fn_data, Ft_data, Veff_data, Alpha_data], axis=0)

    return forces_data, r_centers_norm, azimuth_angles, theta_rad

def prepare_model_data(model_type):
    """Génère X, Y et les splits en fonction du type de modèle."""
    forces_data, r_centers_norm, azimuth_angles, theta_rad = load_base_data()
    
    Fn_data, Ft_data = forces_data[0], forces_data[1]
    Veff_data, Alpha_data = forces_data[2], forces_data[3]

    is_hybrid = model_type in [1, 3]
    is_global = model_type in [2, 3]

    if not is_global:
        R_grid, T_grid = np.meshgrid(r_centers_norm, theta_rad)
        X = np.stack([R_grid.flatten(), np.cos(T_grid).flatten(), np.sin(T_grid).flatten()], axis=1)
        Y_f = np.stack([Fn_data.flatten(), Ft_data.flatten()], axis=1)
        Y_a = np.stack([Veff_data.flatten(), Alpha_data.flatten()], axis=1)
        in_feats, out_feats = 3, 2
    else: 
        X = np.stack([np.cos(theta_rad), np.sin(theta_rad)], axis=1)
        Y_f = np.concatenate([Fn_data, Ft_data], axis=1)
        Y_a = np.concatenate([Veff_data, Alpha_data], axis=1)
        in_feats, out_feats = 2, 68

    Y_target = Y_a if is_hybrid else Y_f
    X_train_full, X_test, Y_train_full, Y_test, Yf_train_full, Yf_test = train_test_split(
        X, Y_target, Y_f, test_size=0.3, random_state=42)

    scaler_Y = StandardScaler()
    Y_train_full_scaled = scaler_Y.fit_transform(Y_train_full)

    # On regroupe tout dans un dictionnaire pour faciliter le transport
    data_ctx = {
        'X': X, 'Y_f': Y_f, 'Y_target': Y_target,
        'X_train_full': X_train_full, 'X_test': X_test,
        'Y_train_full': Y_train_full, 'Y_test': Y_test,
        'Yf_train_full': Yf_train_full, 'Yf_test': Yf_test,
        'Y_train_full_scaled': Y_train_full_scaled,
        'scaler_Y': scaler_Y,
        'in_feats': in_feats, 'out_feats': out_feats,
        'forces_data': forces_data, 'r_centers_norm': r_centers_norm,
        'azimuth_angles': azimuth_angles, 'theta_rad': theta_rad,
        'is_hybrid': is_hybrid, 'is_global': is_global
    }
    return data_ctx

def format_time(seconds):
    """Formate les secondes en un format lisible (Heures, Minutes, Secondes)."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{int(h)}h {int(m)}m {s:.0f}s"
    elif m > 0:
        return f"{int(m)}m {s:.0f}s"
    else:
        return f"{s:.1f}s"

def save_hyperparameters(params, cv_loss, model_type, activation, folder, exec_time):
    """Enregistre les hyperparamètres et le temps d'exécution dans un fichier texte."""
    filename = f"best_params_model_{model_type}_{activation}.txt"
    filepath = os.path.join(folder, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"=== Résultats Optuna - Modèle {model_type} ({activation}) ===\n")
        f.write(f"Best CV MSE Loss : {cv_loss:.6f}\n")
        f.write(f"Temps Optuna     : {format_time(exec_time)}\n")
        f.write("-" * 40 + "\n")
        for k, v in params.items():
            f.write(f"{k: <20}: {v}\n")
        f.write(f"{'activation': <20}: {activation}\n")