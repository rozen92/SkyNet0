# SkyNet0

**SkyNet0** vise à prédire les efforts aérodynamiques s'appliquant sur les pales d'une éolienne à l'aide de réseaux de neurones de type **Multi-Layer Perceptron (MLP)**. 

Contrairement aux autres modèles de la lignée SkyNet qui sont "informés" par la physique, cet algorithme n'utilise pas la théorie de la **BEM (Blade Element Momentum)** pour ses prédictions, privilégiant une approche purement statistiques.

---

## Données : `dataset_forces_mexico.pt`

Le fichier `dataset_forces_mexico.pt` est un tenseur PyTorch contenant les données extraites des simulations SVEN pour l'expérience **New Mexico**. 

Ce jeu de données est focalisé sur un point de fonctionnement spécifique :
* **Yaw :** 30°.
* **TSR (Tip Speed Ratio) :** 12.

Le tenseur contient 4 grandeurs physiques réparties sur **34 sections** de pale et **72 angles azimutaux** (échantillonnage tous les 5°) :
1.  **$F_n$** : Force normale.
2.  **$F_t$** : Force tangentielle.
3.  **$V_{eff}$** : Vitesse effective vue par le profil.
4.  **$\alpha$** : Angle d'attaque.

---

## Les Différents Modèles

Le script permet de tester quatre architectures distinctes pour comparer l'efficacité des approches locales vs globales et directes vs hybrides.

| Modèle | Type | Entrées $\rightarrow$ Sorties | Description |
| :--- | :--- | :--- | :--- |
| **0** | **Direct Local** | $(r, \cos\theta, \sin\theta) \rightarrow (F_n, F_t)$ | Prédit les forces localement pour une section et un angle donnés. |
| **1** | **Hybride Local** | $(r, \cos\theta, \sin\theta) \rightarrow (V_{eff}, \alpha)$ | Prédit l'aérodynamique locale, puis calcule les forces via la loi physique. |
| **2** | **Direct Global** | $(\cos\theta, \sin\theta) \rightarrow (34 \times F_n, 34 \times F_t)$ | Prédit l'ensemble des efforts sur toute la pale simultanément. |
| **3** | **Hybride Global** | $(\cos\theta, \sin\theta) \rightarrow (34 \times V_{eff}, 34 \times \alpha)$ | Approche globale utilisant les paramètres aérodynamiques comme cibles intermédiaires. |

---

## Optimisation et Baseline

### Optuna 
Le projet utilise **Optuna** pour automatiser la recherche des meilleures configurations. Pour chaque modèle et chaque fonction d'activation (**ReLU, Tanh, Sine**), une **validation croisée (K-Fold)** est effectuée pour optimiser le nombre de couches, d'unités, le taux de Dropout et le Learning Rate.

### Interpolateur Baseline
Afin de juger de la performance de l'IA, le script génère systématiquement une **baseline par interpolation linéaire** (`LinearNDInterpolator`). Cela permet de vérifier si le MLP apporte une réelle valeur ajoutée par rapport à une méthode mathématique classique.

---

## Installation et Lancement

### 1. Prérequis

Clonez le dépôt et installez les dépendances nécessaires via le fichier `requirements.txt` :

```bash
# Clonage du dépôt
git clone https://github.com/rozen92/SkyNet0.git
cd SkyNet0

# Installation des bibliothèques
pip install -r requirements.txt
```

---

### 2. Exécution de la campagne de tests

Le script principal lance automatiquement une série de **12 simulations**
*(4 types de modèles × 3 fonctions d’activation)*.

Depuis la racine du projet, exécutez :

```bash
python SkyNet0/newMexico/main.py
```

---

### 3. Résultats

Après l'exécution, les dossiers suivants sont générés dans `newMexico/` :

* **`images/`** : Graphiques de convergence et prédictions azimutales (`.png`)
* **`hyperparametres/`** : Logs des meilleurs paramètres et temps d'exécution (`.txt`)


