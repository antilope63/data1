import pandas as pd
from geopy.distance import geodesic
import numpy as np


def readParquet(file):
    return pd.read_parquet(file)


def list_columns(file):
    df = readParquet(file)
    columns = df.columns.tolist()
    print("Liste des colonnes disponibles avec 3 exemples de valeurs :")
    for col in columns:
        print(f"\nColonne: {col}")
        print(f"Exemples de valeurs: {df[col].dropna().head(3).tolist()}")
    return columns


def get_race_cities(circuits_file):
    circuits = pd.read_csv(circuits_file)
    race_cities = circuits["location"].unique()
    return race_cities


def get_race_info(races_file, circuits_file):
    races = pd.read_csv(races_file)
    circuits = pd.read_csv(circuits_file)
    circuit_locations = circuits.set_index("circuitId")["location"].to_dict()
    races["date"] = pd.to_datetime(races["date"])
    races["location"] = races["circuitId"].map(circuit_locations)

    return races[["raceId", "location", "date"]]


def filter_weather_by_race_info(weather_file, races_info, output_file):
    weather_data = pd.read_parquet(weather_file)
    weather_data["date"] = pd.to_datetime(weather_data["date"])
    filtered_weather = pd.DataFrame()
    for _, race in races_info.iterrows():
        location = race["location"]
        race_date = race["date"]
        weather_for_race = weather_data[
            (weather_data["city_name"] == location)
            & (weather_data["date"] == race_date)
        ]
        print(f"Récupération de la météo pour {location} à la date {race_date}")
        filtered_weather = pd.concat([filtered_weather, weather_for_race])
    filtered_weather.to_csv(output_file, index=False)
    print(f"Les données météo filtrées ont été sauvegardées dans {output_file}")


races_info = get_race_info("races.csv", "circuits.csv")

# Filtrage des données météo et sauvegarde
filter_weather_by_race_info(
    "daily_weather.parquet", races_info, "filtered_weather.csv"
)


# print(readParquet("daily_weather.parquet"))
# list_columns("daily_weather.parquet")


def check_locations_in_weather(circuits_file, weather_file):
    circuits = pd.read_csv(circuits_file)
    unique_locations = circuits["location"].unique()

    weather_data = pd.read_parquet(weather_file)
    weather_cities = weather_data["city_name"].unique()

    missing_locations = []
    for location in unique_locations:
        if location not in weather_cities:
            missing_locations.append(location)
            print(f"Ville manquante dans le fichier météo : {location}")

    print(f"Nombre de villes manquantes dans {weather_file} : {len(missing_locations)}")

    return missing_locations


# # Vérification des villes dans deux fichiers météo différents
# a = check_locations_in_weather("circuits.csv", "archive-2/daily_weather.parquet")
# b = check_locations_in_weather("circuits.csv", "daily_weather.parquet")

# # Concaténer les deux listes de villes manquantes
# c = list(set(a + b))

# print(f"Nombre total de villes manquantes dans les deux fichiers : {len(c)}")
# print(f"Villes manquantes uniques : {c}")
