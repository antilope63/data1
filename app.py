import streamlit as st
import pandas as pd
import joblib
import os

# Chargement du modèle
model = joblib.load("race_predictor_model.pkl")

# Chargement de l'encodeur
le_constructor = joblib.load("le_constructor.joblib")

# Chargement des données nécessaires
circuits = pd.read_csv("circuits.csv")
constructors = pd.read_csv("constructors.csv")

# Déterminer l'extension du fichier image
image_extension = ".jpg"  # ou '.png'

# Ajouter le chemin de l'image au DataFrame
circuits["image_path"] = "images/" + circuits["circuitRef"] + image_extension

# Récupérer les constructorId connus du LabelEncoder
constructor_ids = le_constructor.classes_.astype(int)

# Filtrer le DataFrame des constructors pour ne garder que ceux connus
constructors = constructors[constructors["constructorId"].isin(constructor_ids)]

# Obtenir les noms des écuries correspondantes
constructor_names = constructors["name"].values

# Encoder les constructorId
constructor_ids_enc = le_constructor.transform(constructors["constructorId"].values)

# Titre de l'application
st.title("Simulateur de F1 avec Conditions Météorologiques")

# Sélection du circuit
circuit_names = circuits["name"].unique()
selected_circuit = st.selectbox("Sélectionnez un circuit", circuit_names)

# Récupération des informations du circuit sélectionné
circuit_info = circuits[circuits["name"] == selected_circuit].iloc[0]

# Affichage de l'image du circuit
image_path = circuit_info["image_path"]
if os.path.exists(image_path):
    st.image(image_path, caption=selected_circuit, use_column_width=True)
else:
    st.write("Image non disponible pour ce circuit.")

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

# Bouton pour lancer la simulation
if st.button("Lancer la simulation"):
    # Préparation des données pour le modèle
    # Créer un DataFrame avec les données pour chaque écurie
    input_data = pd.DataFrame(
        {
            "grid": [1]
            * len(
                constructor_ids_enc
            ),  # Vous pouvez ajuster les positions de départ si nécessaire
            "circuitId": [circuit_info["circuitId"]] * len(constructor_ids_enc),
            "year": [2024]
            * len(
                constructor_ids_enc
            ),  # Vous pouvez permettre à l'utilisateur de sélectionner l'année
            "round": [1] * len(constructor_ids_enc),  # Idem pour le numéro de la course
            "lat": [circuit_info["lat"]] * len(constructor_ids_enc),
            "lng": [circuit_info["lng"]] * len(constructor_ids_enc),
            "avg_temp_c": [avg_temp_c] * len(constructor_ids_enc),
            "precipitation_mm": [precipitation_mm] * len(constructor_ids_enc),
            "avg_wind_speed_kmh": [avg_wind_speed_kmh] * len(constructor_ids_enc),
            "constructorId_enc": constructor_ids_enc,
        }
    )

    # Prédiction
    predictions = model.predict(input_data)

    # Préparation des résultats
    results = pd.DataFrame({"Écurie": constructor_names, "Points Prévus": predictions})

    # Trier les résultats par points décroissants
    results = results.sort_values(by="Points Prévus", ascending=False)

    # Affichage des résultats
    st.subheader("Résultats de la Simulation")
    st.write(results.reset_index(drop=True))
