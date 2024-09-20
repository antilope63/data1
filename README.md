# Simulateur de F1 avec Conditions Météorologiques

Bienvenue dans le **Simulateur de F1** ! Cette application vous permet de simuler des courses de Formule 1 en prenant en compte les conditions météorologiques, le circuit choisi, et les pilotes participants pour une année spécifique.

## 🎯 Description

Cette application interactive, construite avec **Streamlit**, prédit les résultats d'une course de F1 en fonction de plusieurs facteurs :

- **Année de la saison**
- **Circuit**
- **Conditions météorologiques**

Le modèle prédit le classement final des pilotes, attribue les points selon le barème officiel de la F1, et génère également le classement des écuries en fonction des points cumulés de leurs pilotes.

## 🛠️ Installation

### Prérequis

- Python 3.7 ou supérieur
- **pip** pour l'installation des packages
- Les packages Python suivants :
  - **streamlit**
  - **pandas**
  - **numpy**
  - **scikit-learn**
  - **joblib**

### Étapes d'installation

1. **Cloner le dépôt ou télécharger les fichiers**

   ```bash
   git clone https://github.com/antilope63/data1
   ```

2. **Naviguer dans le répertoire du projet**

   ```bash
   cd data1
   ```

3. **Installer les dépendances**

   ```bash
   pip install -r requirements.txt
   ```

4. **Entraîner le modèle** (si ce n'est pas déjà fait)

   Exécutez le script `simulateur.py` pour entraîner le modèle et générer les encodeurs :

   ```bash
   python simulateur.py

   ```

   Cela va créer les fichiers suivants :

   - `race_predictor_model.pkl`
   - `le_constructor.joblib`
   - `le_driver.joblib`

## 🚀 Utilisation

1. **Lancer l'application Streamlit**

   ```bash
   streamlit run app.py
   ```

2. **Interagir avec l'application**
   - **Sélectionnez une année**
   - **Choisissez un circuit**
   - **Ajustez les conditions météorologiques**
   - **Lancez la simulation** : Cliquez sur le bouton **"Lancer la simulation"** pour obtenir les résultats.
3. **Interpréter les résultats**
   - **Résultats de la Simulation - Pilotes** : Vous verrez le classement des pilotes avec leur position prédite, leur écurie, les points attribués et le pourcentage de confiance du modèle.
   - **Classement des Écuries** : Le classement des écuries basé sur les points cumulés de leurs pilotes.

## 🤖 Modèle de Prédiction

Le modèle de prédiction est un **Random Forest Regressor** qui prédit la position finale des pilotes en fonction de :

- Position sur la grille de départ
- Circuit
- Année et numéro de la course
- Latitude et longitude du circuit
- Conditions météorologiques (température, précipitations, vent)
- Écurie du pilote
- Identifiant du pilote

## 📌 Remarques Importantes

- Les prédictions sont basées sur des données historiques et des modèles statistiques. Elles ne reflètent pas nécessairement les performances réelles actuelles des pilotes ou des écuries.
- Le pourcentage de confiance est une estimation de l'incertitude du modèle pour chaque prédiction. Il est calculé en fonction de la variance des prédictions des arbres individuels du Random Forest.

##

Merci d'avoir utilisé le **Simulateur de F1** ! 🚗🏁
