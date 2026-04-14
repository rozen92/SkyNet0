# SkyNet0

**SkyNet0** vise à prédire les efforts aérodynamiques s'appliquant sur les pales d'une éolienne à l'aide de réseaux de neurones de type **Multi-Layer Perceptron (MLP)**. 

Contrairement aux autres modèles de la lignée SkyNet qui sont "informés" par la physique, cet algorithme n'utilise pas la théorie de la **BEM (Blade Element Momentum)** pour ses prédictions, privilégiant une approche purement statistiques.

---

## Données : `dataset_forces_mexico.xlsx`

Le fichier `dataset_forces_mexico.xlsx` regroupe les données extraites des simulations SVEN pour l'expérience **New Mexico**. 

### 1. Points de fonctionnement inclus
Le jeu de données couvre une matrice de conditions opérationnelles permettant d'étudier l'impact du lacet et de la charge :
* **Yaw (Lacet) :** 15° et 30°.
* **TSR (Tip Speed Ratio) :** 4, 8 et 12.

### 2. Discrétisation Spatiale et Temporelle
* **Rayon ($r$) :** 34 sections discrétisées le long de la pale (du moyeu au bout de pale).
* **Azimut ($\theta$) :** 36 points (échantillonnage tous les 10°, de 0° à 350°).

### 3. Structure du fichier (Colonnes)
Chaque ligne représente un point de calcul $(r, \theta)$ pour un couple (Yaw, TSR) donné :

| Colonne | Description | Unité |
| :--- | :--- | :--- |
| `r` | Position radiale locale | [m] |
| `theta` | Angle azimutal (0° à 9H, sens horaire) | [°] |
| `yaw` | Angle de lacet de la turbine | [°] |
| `TSR` | Ratio de vitesse en bout de pale | [-] |
| `Fn` | Force normale  | [N/m] |
| `Ft` | Force tangentielle | [N/m] |
| `V_eff` | Vitesse effective vue par le profil | [m/s] |
| `alpha` | Angle d'attaque local | [°] |
| `err_period_n` | Amplitude des fluctuations de $F_n$ sur un tour | [N/m] |
| `err_period_t` | Amplitude des fluctuations de $F_t$ sur un tour | [N/m] |

> **Note technique :** Le code `SkyNet0` filtre ces données via `utils.py` pour s'entraîner par défaut sur le point de fonctionnement **Yaw 15° / TSR 8**.

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
Afin de juger de la performance de l'IA, le script génère systématiquement une **baseline par interpolation linéaire** (`LinearNDInterpolator`). Cela permet de vérifier si le MLP apporte une réelle valeur ajoutée par rapport à une méthode mathématique basique.

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


