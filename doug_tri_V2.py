import pandas as pd


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
# filter_weather_by_race_info("daily_weather.parquet", races_info, "filtered_weather.csv")


# print(readParquet("daily_weather.parquet"))
# list_columns("daily_weather.parquet")


import pandas as pd


def get_missing_weather_for_cities(
    circuits_file, cities_file, weather_file, output_file
):
    # Charger les fichiers
    circuits = pd.read_csv(circuits_file)
    cities = pd.read_csv(cities_file)
    weather_data = pd.read_parquet(weather_file)

    # Initialiser un DataFrame pour stocker les nouvelles données météo
    new_weather_data = []

    # Parcourir chaque ville manquante dans circuits.csv
    for index, circuit in circuits.iterrows():
        circuit_city = circuit["location"]
        circuit_country = circuit["country"]

        # Trouver toutes les villes dans le même pays
        cities_in_country = cities[cities["country"] == circuit_country][
            "city_name"
        ].unique()

        # Récupérer la météo pour ces villes
        weather_for_country = weather_data[
            weather_data["city_name"].isin(cities_in_country)
        ]

        # Si des données météo sont trouvées, calculer la moyenne
        if not weather_for_country.empty:
            avg_weather = weather_for_country.mean(numeric_only=True).round(
                1
            )  # Arrondir à 1 chiffre après la virgule
            print(
                f"Moyenne des données météo pour {circuit_city} (basée sur les villes environnantes dans {circuit_country}) :\n{avg_weather}"
            )

            # Ajouter la localisation (city_name) au lieu de country
            avg_weather["city_name"] = circuit_city  # Ajouter le nom de la ville

            # Stocker les nouvelles données météo
            new_weather_data.append(avg_weather)
        else:
            print(
                f"Aucune donnée météo trouvée pour les villes de {circuit_country}. Ignoré."
            )

    # Si des nouvelles données sont disponibles, les ajouter au fichier de sortie
    if new_weather_data:
        new_weather_df = pd.DataFrame(new_weather_data)

        # Charger le fichier filtered_weather.csv s'il existe déjà
        try:
            filtered_weather = pd.read_csv(output_file)
        except FileNotFoundError:
            filtered_weather = pd.DataFrame()

        # Concaténer les nouvelles données avec les anciennes
        combined_weather_data = pd.concat(
            [filtered_weather, new_weather_df], ignore_index=True
        )

        # Sauvegarder le nouveau fichier avec les données mises à jour
        combined_weather_data.to_csv(output_file, index=False)
        print(
            f"Les nouvelles données météo ont été ajoutées et sauvegardées dans {output_file}"
        )
    else:
        print("Aucune nouvelle donnée à ajouter.")


# Utilisation du code
get_missing_weather_for_cities(
    "circuits.csv", "cities.csv", "daily_weather.parquet", "filtered_weather.csv"
)
