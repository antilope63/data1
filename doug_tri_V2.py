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


# Fonction pour charger et récupérer les villes où il y a eu des courses
def get_race_cities(circuits_file):
    circuits = pd.read_csv(circuits_file)
    race_cities = circuits["location"].unique()
    return race_cities


def get_race_cities_and_dates(races_file, circuits_file):
    races = pd.read_csv(races_file)
    circuits = pd.read_csv(circuits_file)

    races_with_cities = pd.merge(races, circuits, on="circuitId")
    race_cities_dates = races_with_cities[["location", "date"]]

    race_cities_dates.loc[:, "date"] = pd.to_datetime(race_cities_dates["date"])

    return race_cities_dates


def get_weather_for_races(weather_file, race_cities_dates, output_file):
    weather_data = pd.read_parquet(weather_file)
    weather_data["date"] = pd.to_datetime(weather_data["date"])

    weather_for_races = []

    for _, race in race_cities_dates.iterrows():
        # Filtrer les données météo pour la ville et la date de la course
        weather_match = weather_data[
            (weather_data["city_name"] == race["location"])
            & (weather_data["date"] == race["date"])
        ]

        # Ajouter les données filtrées à la liste des résultats
        if not weather_match.empty:
            print(
                f"Récupération de la météo pour la course à {race['location']} le {race['date'].date()}"
            )
            weather_for_races.append(weather_match)

    # Concaténer toutes les données météo récupérées
    final_weather_data = pd.concat(weather_for_races, ignore_index=True)

    # Sauvegarder les résultats dans un fichier CSV
    final_weather_data.to_csv(output_file, index=False)
    print(f"Les données météo pour les courses ont été sauvegardées dans {output_file}")


race_cities_dates = get_race_cities_and_dates("races.csv", "circuits.csv")
get_weather_for_races(
    "daily_weather.parquet",
    race_cities_dates,
    "filtered_weather_by_race_cities_and_dates.csv",
)


# print(readParquet("daily_weather.parquet"))
# list_columns("daily_weather.parquet")
