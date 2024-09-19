import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Chargement du modèle
model = joblib.load("race_predictor_model.pkl")

# Chargement des encodeurs
le_constructor = joblib.load("le_constructor.joblib")
le_driver = joblib.load("le_driver.joblib")

# Chargement des données nécessaires
circuits = pd.read_csv("circuits.csv")
drivers = pd.read_csv("drivers.csv")
results_data = pd.read_csv("results.csv")
constructors = pd.read_csv("constructors.csv")
races = pd.read_csv("races.csv")

# Titre de l'application
st.title("Simulateur de F1 avec Conditions Météorologiques")

# Sélection de l'année
st.subheader("Sélection de l'Année")
year_options = sorted(races["year"].unique())
selected_year = st.selectbox("Sélectionnez une année", year_options)

# Sélection du circuit
circuit_names = circuits["name"].unique()
selected_circuit = st.selectbox("Sélectionnez un circuit", circuit_names)

# Récupération des informations du circuit sélectionné
circuit_info = circuits[circuits["name"] == selected_circuit].iloc[0]

# Affichage de l'image du circuit (si applicable)
image_path = circuit_info.get("image_path", None)
if image_path and os.path.exists(image_path):
    st.image(image_path, caption=selected_circuit, use_column_width=True)

# Affichage des informations du circuit
st.write(f"**Lieu** : {circuit_info['location']}, {circuit_info['country']}")
st.write(f"**Latitude** : {circuit_info['lat']}")
st.write(f"**Longitude** : {circuit_info['lng']}")

# Sélection des conditions météorologiques
st.subheader("Conditions Météorologiques")
avg_temp_c = st.slider(
    "Température moyenne (°C)", min_value=-10, max_value=40, value=20
)
precipitation_mm = st.slider("Précipitations (mm)", min_value=0, max_value=50, value=0)
avg_wind_speed_kmh = st.slider(
    "Vitesse moyenne du vent (km/h)", min_value=0, max_value=100, value=10
)

# Récupérer les races de l'année sélectionnée
races_selected_year = races[races["year"] == selected_year]
race_ids = races_selected_year["raceId"].unique()

# Filtrer les résultats pour ces raceId
results_selected_year = results_data[results_data["raceId"].isin(race_ids)]

# Récupérer les driverId et constructorId
driver_ids = results_selected_year["driverId"].unique()
constructor_ids = results_selected_year["constructorId"].unique()

# Filtrer les driverId et constructorId connus du modèle
known_driver_ids = le_driver.classes_.astype(int)
driver_ids = np.array([id for id in driver_ids if id in known_driver_ids])

known_constructor_ids = le_constructor.classes_.astype(int)
constructor_ids = np.array(
    [id for id in constructor_ids if id in known_constructor_ids]
)

# Filtrer les pilotes et écuries
drivers = drivers[drivers["driverId"].isin(driver_ids)]
constructors = constructors[constructors["constructorId"].isin(constructor_ids)]

# Associer les pilotes à leurs écuries
driver_constructor = results_selected_year[
    ["driverId", "constructorId"]
].drop_duplicates()
drivers = drivers.merge(driver_constructor, on="driverId", how="left")

# Convertir 'constructorId' en entier
drivers["constructorId"] = drivers["constructorId"].astype(int)

# Filtrer les pilotes pour ne garder que ceux dont le 'constructorId' est connu
drivers = drivers[drivers["constructorId"].isin(known_constructor_ids)]

# Mettre à jour 'driver_ids' et 'constructor_ids' après le filtrage
driver_ids = drivers["driverId"].astype(int).values
constructor_ids = drivers["constructorId"].astype(int).values

# Encoder les 'driverId' et 'constructorId'
driver_ids_enc = le_driver.transform(driver_ids)
constructor_ids_enc = le_constructor.transform(constructor_ids)

# Obtenir les noms des pilotes et des écuries
driver_names = drivers["forename"] + " " + drivers["surname"]
constructor_names = (
    constructors.set_index("constructorId").loc[constructor_ids]["name"].values
)

# Afficher le nombre de pilotes disponibles
st.write(f"Nombre de pilotes disponibles pour la simulation : {len(driver_ids)}")

# Bouton pour lancer la simulation
if st.button("Lancer la simulation"):
    # Préparation des données pour le modèle
    num_drivers = len(driver_ids_enc)

    # Générer des positions de départ moyennes
    grid_positions_data = (
        results_selected_year.groupby("driverId")["grid"].mean().reset_index()
    )
    grid_positions_data = drivers[["driverId"]].merge(
        grid_positions_data, on="driverId", how="left"
    )
    grid_positions = (
        grid_positions_data["grid"]
        .fillna(grid_positions_data["grid"].mean())
        .astype(int)
        .values
    )

    input_data = pd.DataFrame(
        {
            "grid": grid_positions,
            "circuitId": [circuit_info["circuitId"]] * num_drivers,
            "year": [selected_year] * num_drivers,
            "round": [1] * num_drivers,
            "lat": [circuit_info["lat"]] * num_drivers,
            "lng": [circuit_info["lng"]] * num_drivers,
            "avg_temp_c": [avg_temp_c] * num_drivers,
            "precipitation_mm": [precipitation_mm] * num_drivers,
            "avg_wind_speed_kmh": [avg_wind_speed_kmh] * num_drivers,
            "constructorId_enc": constructor_ids_enc,
            "driverId_enc": driver_ids_enc,
        }
    )

    # Réordonner les colonnes selon l'ordre des features du modèle
    input_data = input_data[model.feature_names]

    # Prédiction de la position finale
    predicted_positions = model.predict(input_data)

    # Éviter les égalités en ajoutant un très petit bruit aléatoire
    np.random.seed(42)
    predicted_positions += np.random.normal(0, 0.0001, size=predicted_positions.shape)

    # Arrondir les positions prédites et s'assurer qu'elles sont dans un intervalle valide
    predicted_positions = np.round(predicted_positions).astype(int)
    predicted_positions = np.clip(predicted_positions, 1, 20)

    # Calculer les scores de confiance
    all_tree_predictions = np.array(
        [tree.predict(input_data) for tree in model.estimators_]
    )
    std_predictions = np.std(all_tree_predictions, axis=0)
    max_std = std_predictions.max() if std_predictions.max() != 0 else 1
    confidence_scores = (1 - (std_predictions / max_std)) * 100
    confidence_scores = np.round(confidence_scores, 2)

    # Créer un DataFrame avec les résultats
    results = pd.DataFrame(
        {
            "Pilote": driver_names,
            "Écurie": constructor_names,
            "Position Prédite": predicted_positions,
            "Confiance (%)": confidence_scores,
        }
    )

    # Attribuer les points selon le barème de la F1
    points_distribution = {
        1: 25,
        2: 18,
        3: 15,
        4: 12,
        5: 10,
        6: 8,
        7: 6,
        8: 4,
        9: 2,
        10: 1,
    }
    results["Points Attribués"] = (
        results["Position Prédite"].map(points_distribution).fillna(0).astype(int)
    )

    # Trier les résultats par position prédite croissante
    results = results.sort_values(by="Position Prédite")

    # Réordonner les colonnes
    results = results[
        ["Position Prédite", "Pilote", "Écurie", "Points Attribués", "Confiance (%)"]
    ]

    # Affichage des résultats pour les pilotes
    st.subheader("Résultats de la Simulation - Pilotes")
    st.write(results.reset_index(drop=True))

    # Calculer les points totaux par écurie
    team_points = results.groupby("Écurie")["Points Attribués"].sum().reset_index()

    # Trier les écuries par points décroissants
    team_points = team_points.sort_values(by="Points Attribués", ascending=False)

    # Affichage des résultats pour les écuries
    st.subheader("Classement des Écuries")
    st.write(team_points.reset_index(drop=True))
