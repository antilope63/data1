import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ------------------------------ Préparation des données ------------------------------

circuits = pd.read_csv("circuits.csv")
races = pd.read_csv("races.csv")
results = pd.read_csv("results.csv")
weather = pd.read_csv("filtered_weather.csv")
drivers = pd.read_csv("drivers.csv")

# -------------------------------- Fusion des données --------------------------------

race_circuit = pd.merge(races, circuits, on="circuitId")
race_results = pd.merge(results, race_circuit, on="raceId")
race_results["date"] = pd.to_datetime(race_results["date"])
weather["date"] = pd.to_datetime(weather["date"])
data = pd.merge(race_results, weather, on=["raceId", "date"], how="left")
data.rename(columns={"circuitId_x": "circuitId"}, inplace=True)
data.drop(columns=["circuitId_y"], inplace=True, errors="ignore")

data_model = data[
    [
        "raceId",
        "driverId",
        "constructorId",
        "grid",
        "positionOrder",
        "points",
        "circuitId",
        "year",
        "round",
        "lat",
        "lng",
        "avg_temp_c",
        "precipitation_mm",
        "avg_wind_speed_kmh",
    ]
]

data_model = data_model.dropna()

le_constructor = LabelEncoder()
data_model["constructorId_enc"] = le_constructor.fit_transform(
    data_model["constructorId"]
)

le_driver = LabelEncoder()
data_model["driverId_enc"] = le_driver.fit_transform(data_model["driverId"])

features = data_model[
    [
        "grid",
        "circuitId",
        "year",
        "round",
        "lat",
        "lng",
        "avg_temp_c",
        "precipitation_mm",
        "avg_wind_speed_kmh",
        "constructorId_enc",
        "driverId_enc",
    ]
]

target = data_model["positionOrder"]

# -------------------------- Entraînement du Modèle de Prédiction --------------------------

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# Création du modèle
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Entraînement du modèle
model.fit(X_train, y_train)

# Sauvegarder les noms des features
model.feature_names = features.columns.tolist()


y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Erreur absolue moyenne (MAE) : {mae}")

# Sauvegarde du modèle entraîné
joblib.dump(model, "race_predictor_model.pkl")
joblib.dump(le_constructor, "le_constructor.joblib")
joblib.dump(le_driver, "le_driver.joblib")
