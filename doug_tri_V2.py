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


def get_missing_weather_circuits(
    circuits_file, weather_file, cities_file, countries_file, output_file
):
    circuits = pd.read_csv(circuits_file)
    weather_data = pd.read_parquet(weather_file)
    cities = pd.read_csv(cities_file)
    countries = pd.read_csv(countries_file)

    unique_locations = circuits["location"].unique()
    weather_cities = weather_data["city_name"].unique()
    missing_locations = [
        location for location in unique_locations if location not in weather_cities
    ]

    print(f"Villes sans données météo : {missing_locations}")

    new_weather_data = []

    for missing_location in missing_locations:
        circuit_country = circuits[circuits["location"] == missing_location][
            "country"
        ].values[0]
        print(f"Récupération des données pour le pays : {circuit_country}")

        cities_in_country = cities[cities["country"] == circuit_country][
            "city_name"
        ].unique()
        country_weather_data = weather_data[
            weather_data["city_name"].isin(cities_in_country)
        ]

        if not country_weather_data.empty:
            avg_weather = country_weather_data.mean(numeric_only=True)
            print(
                f"Moyenne des données météo pour {missing_location} (basée sur les villes environnantes) :\n{avg_weather}"
            )

            # Ajouter les informations du circuit et de la moyenne météo dans le DataFrame
            avg_weather["location"] = missing_location
            avg_weather["country"] = circuit_country

            # Ajouter ces nouvelles données météo au fichier
            new_weather_data.append(avg_weather)
        else:
            print(
                f"Aucune donnée météo trouvée pour les villes du pays {circuit_country}. Ignoré."
            )

    # Si des données ont été calculées, on les ajoute au fichier principal
    if new_weather_data:
        new_weather_df = pd.DataFrame(new_weather_data)
        combined_weather_data = pd.concat(
            [weather_data, new_weather_df], ignore_index=True
        )

        # Sauvegarder le fichier avec les nouvelles données
        combined_weather_data.to_csv(output_file, index=False)
        print(
            f"Les nouvelles données météo ont été ajoutées et sauvegardées dans {output_file}"
        )
    else:
        print("Aucune nouvelle donnée à ajouter.")

    print("Traitement terminé.")


# Utilisation du code
# get_missing_weather_circuits(
#     "circuits.csv",
#     "daily_weather.parquet",
#     "cities.csv",
#     "countries.csv",
#     "weather_data_with_estimates.csv",
# )
