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


# j'arrive pas à acceder à aux saisons
def get_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"


def get_race_info(races_file, circuits_file):
    races = pd.read_csv(races_file)
    circuits = pd.read_csv(circuits_file)
    races_with_circuits = pd.merge(races, circuits, on="circuitId")
    races_info = races_with_circuits[["circuitId", "location", "date", "country"]]
    races_info.loc[:, "date"] = pd.to_datetime(races_info["date"])

    return races_info


def filter_weather_by_race_info(weather_file, races_info, cities_file, output_file):
    weather_data = pd.read_parquet(weather_file)
    weather_data["date"] = pd.to_datetime(weather_data["date"])
    cities = pd.read_csv(cities_file)

    filtered_weather = pd.DataFrame()

    for _, race in races_info.iterrows():
        location = race["location"]
        race_date = race["date"]
        circuit_country = race["country"]
        weather_for_race = weather_data[
            (weather_data["city_name"] == location)
            & (weather_data["date"] == race_date)
        ]

        if not weather_for_race.empty:
            print(f"Météo trouvée pour {location} à la date {race_date}")
            filtered_weather = pd.concat([filtered_weather, weather_for_race])
        else:
            print(
                f"Pas de météo pour {location} à la date {race_date}, calcul de la moyenne..."
            )
            cities_in_country = cities[cities["country"] == circuit_country][
                "city_name"
            ].unique()
            weather_for_country = weather_data[
                (weather_data["city_name"].isin(cities_in_country))
                & (weather_data["date"] == race_date)
            ]

            if not weather_for_country.empty:
                avg_weather = weather_for_country.mean(numeric_only=True).round(1)
                season = get_season(race_date)
                avg_weather["station_id"] = None
                avg_weather["city_name"] = location
                avg_weather["date"] = race_date
                avg_weather["season"] = season
                avg_weather = avg_weather[
                    [
                        "station_id",
                        "city_name",
                        "date",
                        "season",
                        "avg_temp_c",
                        "min_temp_c",
                        "max_temp_c",
                        "precipitation_mm",
                        "snow_depth_mm",
                        "avg_wind_dir_deg",
                        "avg_wind_speed_kmh",
                        "peak_wind_gust_kmh",
                        "avg_sea_level_pres_hpa",
                        "sunshine_total_min",
                    ]
                ]

                filtered_weather = pd.concat(
                    [filtered_weather, pd.DataFrame([avg_weather])]
                )
            else:
                print(
                    f"Aucune donnée météo trouvée pour les villes de {circuit_country} à la date {race_date}. Ignoré."
                )

    filtered_weather.to_csv(output_file, index=False)
    print(f"Les données météo filtrées ont été sauvegardées dans {output_file}")


races_info = get_race_info("races.csv", "circuits.csv")

filter_weather_by_race_info(
    "daily_weather.parquet", races_info, "cities.csv", "filtered_weather.csv"
)
