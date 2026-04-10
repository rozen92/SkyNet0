import os
import torch
import json 
from utils import prepare_model_data, format_time, save_hyperparameters
from optimizer import find_best_hyperparameters
from evaluator import evaluate_mlp, evaluate_interpolation
import time

def main():
    global_start_time = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Lancement de la campagne sur : {device}")

    # Configuration des chemins
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir, 'images')
    hp_dir = os.path.join(base_dir, 'hyperparamètres')
    
    # Création des dossiers
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(hp_dir, exist_ok=True)

    activations = ['ReLU', 'Tanh', 'Sine']
    model_types = [0, 1, 2, 3]

    for m_type in model_types:
        model_start_time = time.time()
        print("\n" + "#"*60)
        print(f" DÉMARRAGE DU MODÈLE TYPE {m_type}")
        print("#"*60)
        
        # 1. Chargement et préparation spécifique au modèle
        step_start = time.time()
        data_ctx = prepare_model_data(m_type)
        print(f" Temps de préparation des données : {format_time(time.time() - step_start)}")

        # 2. Exécution de l'Interpolation Classique (Baseline)
        step_start = time.time()
        evaluate_interpolation(data_ctx, m_type, image_dir)
        print(f" Temps de l'interpolation : {format_time(time.time() - step_start)}")

        # 3. Boucle sur les Réseaux de Neurones
        for act in activations:
            act_start_time = time.time()
            
            # Recherche Optuna
            step_start = time.time()
            best_params, best_cv_loss = find_best_hyperparameters(data_ctx, act, n_trials=100, device=device)
            optuna_time = time.time() - step_start
            
            print("\n" + "="*40 + f"\n MEILLEURS HYPERPARAMÈTRES ({act})\n" + "="*40)
            for k, v in best_params.items(): print(f" {k: <15}: {v}")
            print(f" {'Activation': <15}: {act}\n" + "="*40)
            print(f" Temps Optuna : {format_time(optuna_time)}")

            # Sauvegarde dans le dossier hyperparamètres
            save_hyperparameters(best_params, best_cv_loss, m_type, act, hp_dir, optuna_time)

            # Entraînement final et Visualisation
            step_start = time.time()
            evaluate_mlp(data_ctx, best_params, best_cv_loss, act, m_type, image_dir, device)
            print(f"Temps Entraînement Final & Graphes : {format_time(time.time() - step_start)}")
            
            print(f"Temps total pour le réseau ({act}) : {format_time(time.time() - act_start_time)}")

        print(f"\n TEMPS TOTAL POUR LE MODÈLE {m_type} : {format_time(time.time() - model_start_time)}")


    global_total_time = time.time() - global_start_time
    print("\n" + "*"*60)
    print(f" CAMPAGNE TERMINÉE !")
    print(f" TEMPS GLOBAL D'EXÉCUTION : {format_time(global_total_time)}")
    print(f"Images sauvegardées dans : {image_dir}")
    print(f"Fichiers TXT sauvegardés dans : {hp_dir}")
    print("*"*60)

if __name__ == "__main__":
    main()

