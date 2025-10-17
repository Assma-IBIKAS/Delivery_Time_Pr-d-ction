# 📦 Prédiction du Temps de Livraison

## Contexte du Projet   
Je suis une **data scientist junior** et ce projet a été réalisé pour répondre à un cas pratique proposé sur la prédiction du temps de livraison dans une entreprise de logistique.
 

L’objectif est de **prédire le temps total d’une livraison** (`DeliveryTime`) afin de :  
- Anticiper les retards  
- Informer les clients en temps réel  
- Optimiser l’organisation des tournées  

### Variables principales utilisées
| Variable | Description |
|----------|-------------|
| `Distance_km` | Distance entre le restaurant et l’adresse de livraison |
| `Traffic_Level` | Niveau de trafic |
| `Vehicle_Type` | Type de véhicule utilisé |
| `Time of Day` | Heure de la journée |
| `Courier Experience` | Expérience du livreur |
| `Weather` | Conditions météorologiques |
| `Preparation_time` | Temps de préparation de la commande |

### Situation initiale
- Les estimations de temps de livraison étaient encore faites manuellement  
- Aucun modèle de prédiction fiable n’était disponible  
- Les retards fréquents créaient une insatisfaction client  

---

## 🔍 Mon approche

### 1. Analyse et exploration des données
- Heatmap pour analyser la corrélation entre les variables numériques  
- Countplots pour visualiser la distribution des variables catégorielles  
- Boxplots pour comprendre l’impact des variables catégorielles sur la cible  
- Analyse de la distribution de la variable `DeliveryTime`  

> Cette étape m’a permis d’identifier les variables les plus importantes avant de passer à la modélisation.

### 2. Prétraitement et sélection de features
- Standardisation des variables numériques avec **StandardScaler**  
- Encodage des variables catégorielles avec **OneHotEncoder**  
- Sélection des features les plus pertinentes avec **SelectKBest** et `f_regression`

### 3. Modélisation
- Modèles testés :  
  - `RandomForestRegressor`  
  - `SVR`  
- Hyperparamètres optimisés avec **GridSearchCV**
- Évaluation des modèles sur le jeu de test avec **MAE** et **R²**  
- Choix du modèle final basé sur **la meilleure performance sur MAE et R²**

### 4. Tests automatisés
- Vérification du **format et des dimensions** des données  
- Vérification que la **MAE maximale** ne dépasse pas un seuil défini  


