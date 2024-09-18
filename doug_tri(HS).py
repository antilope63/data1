import pandas as pd
from geopy.distance import geodesic
import numpy as np


def readParquet(file):
    return pd.read_parquet(file)


def readCSV(file):
    return pd.read_csv(file)


def list_columns(file):
    df = readParquet(file)
    columns = df.columns.tolist()
    print("Liste des colonnes disponibles avec 3 exemples de valeurs :")
    for col in columns:
        print(f"\nColonne: {col}")
        print(f"Exemples de valeurs: {df[col].dropna().head(3).tolist()}")
    return columns


def dropColumnsParquet(parquet_file, columns_to_drop):
    df = readParquet(parquet_file)
    df = df.drop(columns=columns_to_drop, errors="ignore")
    df.to_parquet(parquet_file, index=False)
    print(
        f"Les colonnes {columns_to_drop} ont été supprimées du fichier {parquet_file}."
    )
    print(f"Nombre de lignes après suppression : {len(df)}")


def dropColumnsCSV(csv_file, unimportant_columns):
    df = pd.read_csv(csv_file)
    df = df.drop(columns=unimportant_columns, errors="ignore")
    df.to_csv(csv_file, index=False)
    print(f"Colonnes non importantes supprimées dans {csv_file}.")


def list_csv_columns(file_list):
    for file in file_list:
        try:
            df = pd.read_csv(file)
            print(f"Fichier: {file}")
            print("Colonnes :")
            print(df.columns.tolist())
            print("\n")
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier {file}: {e}")


def verify_date_ranges(weather_df, races_df):
    print(
        f"Plage de dates des données météo : de {weather_df['datetime'].min()} à {weather_df['datetime'].max()}"
    )
    print(
        f"Plage de dates des courses : de {races_df['date'].min()} à {races_df['date'].max()}"
    )


def filter_by_date(parquet_file, races_file):
    weather_df = readParquet(parquet_file)
    races_df = readCSV(races_file)
    min_weather_date = pd.to_datetime("2022-06-28")
    max_weather_date = pd.to_datetime("2023-07-30")
    races_df["race_datetime"] = pd.to_datetime(
        races_df["date"] + " " + races_df["time"], errors="coerce"
    )
    races_df = races_df[
        (races_df["race_datetime"] >= min_weather_date)
        & (races_df["race_datetime"] <= max_weather_date)
    ]
    weather_df["datetime"] = pd.to_datetime(
        weather_df["apply_time_rl"], unit="s", errors="coerce"
    )
    filtered_weather_df = weather_df[
        weather_df["datetime"].dt.date.isin(races_df["race_datetime"].dt.date)
    ]
    if not filtered_weather_df.empty:
        filtered_weather_df.to_csv("filtered_weather_by_date.csv", index=False)
        print(
            "Filtrage par date effectué. Fichier sauvegardé sous 'filtered_weather_by_date.csv'."
        )
    else:
        print(
            "Aucune correspondance trouvée entre les dates des courses et les données météo."
        )


def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km


def filter_weather_by_circuit_and_date(
    weather_df, circuits_df, races_df, radius_km=20, time_window_days=1
):
    circuits_df.columns = circuits_df.columns.str.strip()
    races_df.columns = races_df.columns.str.strip()
    race_circuits_df = pd.merge(races_df, circuits_df, on="circuitId", how="inner")
    print("Columns after merge:", race_circuits_df.columns)
    print("Columns in weather_df:", weather_df.columns)
    race_circuits_df["race_datetime"] = pd.to_datetime(
        race_circuits_df["date"] + " " + race_circuits_df["time"], errors="coerce"
    )
    weather_df["weather_datetime"] = pd.to_datetime(
        weather_df["apply_time_rl"], unit="s", errors="coerce"
    )
    filtered_weather = pd.DataFrame()
    weather_lats = weather_df["fact_latitude"].values
    weather_lons = weather_df["fact_longitude"].values
    weather_times = weather_df["weather_datetime"]
    weather_times = pd.to_datetime(weather_times)

    for index, circuit in race_circuits_df.iterrows():
        circuit_lat = circuit["lat"]
        circuit_lng = circuit["lng"]
        circuit_name = circuit.get("name_x") or circuit.get("name_y")
        if not circuit_name:
            print(
                "The 'name' column or its variants do not exist in the merged DataFrame."
            )
            continue

        race_time = circuit["race_datetime"]
        if pd.isnull(race_time):
            print(f"Skipping circuit {circuit_name} due to invalid race time.")
            continue

        print(
            f"Processing circuit: {circuit_name} at ({circuit_lat}, {circuit_lng}) on {race_time}"
        )
        distances = haversine_np(weather_lons, weather_lats, circuit_lng, circuit_lat)
        time_differences = (weather_times - race_time).abs().dt.total_seconds() / 3600
        time_differences = time_differences.astype(int)
        radius_mask = distances <= radius_km
        time_mask = time_differences <= (time_window_days * 24)
        combined_mask = radius_mask & time_mask
        weather_near_circuit = weather_df[combined_mask].copy()
        weather_near_circuit["circuit_name"] = circuit_name
        weather_near_circuit["race_datetime"] = race_time
        filtered_weather = pd.concat([filtered_weather, weather_near_circuit])
    filtered_weather.drop_duplicates(inplace=True)
    filtered_weather.to_csv(
        "filtered_weather_by_date_and_localisation.csv",
        index=False,
        header=True,
    )
    print("Filtered weather data saved successfully.")


def clean_missing_data(parquet_file, important_columns):
    df = readParquet(parquet_file)
    missing_rows = df[df[important_columns].isnull().any(axis=1)]

    if not missing_rows.empty:
        print(f"Lignes supprimées dans {parquet_file} :")
        for index, row in missing_rows.iterrows():
            print(f"Ligne {index} supprimée :")
            print(row)
            print("-" * 50)
    cleaned_df = df.dropna(subset=important_columns)
    cleaned_df.to_parquet(parquet_file, index=False)
    print(
        f"Nettoyage des données manquantes terminé. Fichier {parquet_file} mis à jour."
    )


def clean_missing_data_csv(csv_file, important_columns):
    df = pd.read_csv(csv_file)
    existing_columns = [col for col in important_columns if col in df.columns]
    if not existing_columns:
        print(f"Aucune des colonnes importantes n'est présente dans {csv_file}.")
        return
    missing_rows = df[df[existing_columns].isnull().any(axis=1)]
    if not missing_rows.empty:
        print(f"Lignes supprimées dans {csv_file} :")
        for index, row in missing_rows.iterrows():
            print(f"Ligne {index} supprimée :")
            print(row)
            print("-" * 50)
    cleaned_df = df.dropna(subset=existing_columns)
    cleaned_df.to_csv(csv_file, index=False)
    print(f"Nettoyage des données manquantes terminé. Fichier {csv_file} mis à jour.")


# --------------------------------------- déclarations des variables --------------------------------------


file_list = [
    "circuits.csv",
    "constructor_results.csv",
    "constructor_standings.csv",
    "constructors.csv",
    "lap_times.csv",
    "pit_stops.csv",
    "drivers.csv",
    "driver_standings.csv",
    "qualifying.csv",
    "races.csv",
    "results.csv",
    "seasons.csv",
    "sprint_results.csv",
    "status.csv",
]

columns_to_drop = [
    "cmc_0_0_0_1000",
    "cmc_0_0_0_2",
    "cmc_0_0_0_500",
    "cmc_0_0_0_700",
    "cmc_0_0_0_850",
    "cmc_0_0_0_925",
    "cmc_0_0_7_1000",
    "cmc_0_0_7_500",
    "cmc_0_0_7_700",
    "cmc_0_0_7_850",
    "cmc_0_0_7_925",
    "cmc_0_1_0_0",
    "cmc_0_1_11_0",
    "cmc_0_1_65_0",
    "cmc_0_1_66_0",
    "cmc_0_1_67_0",
    "cmc_0_1_68_0",
    "cmc_0_1_7_0",
    "cmc_0_2_2_10",
    "cmc_0_2_2_1000",
    "cmc_0_2_2_500",
    "cmc_0_2_2_700",
    "cmc_0_2_2_850",
    "cmc_0_2_2_925",
    "cmc_0_2_3_10",
    "cmc_0_2_3_1000",
    "cmc_0_2_3_500",
    "cmc_0_2_3_700",
    "cmc_0_2_3_850",
    "cmc_0_2_3_925",
    "cmc_0_3_0_0",
    "cmc_0_3_0_0_next",
    "cmc_0_3_1_0",
    "cmc_0_3_5_1000",
    "cmc_0_3_5_500",
    "cmc_0_3_5_700",
    "cmc_0_3_5_850",
    "cmc_0_3_5_925",
    "gfs_temperature_10000",
    "gfs_temperature_15000",
    "gfs_temperature_20000",
    "gfs_temperature_25000",
    "gfs_temperature_30000",
    "gfs_temperature_35000",
    "gfs_temperature_40000",
    "gfs_temperature_45000",
    "gfs_temperature_5000",
    "gfs_temperature_50000",
    "gfs_temperature_55000",
    "gfs_temperature_60000",
    "gfs_temperature_65000",
    "gfs_temperature_7000",
    "gfs_temperature_70000",
    "gfs_temperature_75000",
    "gfs_temperature_80000",
    "gfs_temperature_85000",
    "gfs_temperature_90000",
    "gfs_temperature_92500",
    "gfs_temperature_95000",
    "gfs_temperature_97500",
    "gfs_total_clouds_cover_high",
    "gfs_total_clouds_cover_middle",
    "gfs_temperature_sea",
    "gfs_temperature_sea_grad",
    "gfs_temperature_sea_interpolated",
    "gfs_temperature_sea_next",
    "gfs_timedelta_s",
    "topography_bathymetry",
]

important_columns = [
    "climate_temperature",
    "gfs_humidity",
    "gfs_pressure",
    "gfs_wind_speed",
    "gfs_precipitations",
    "gfs_cloudness",
    "gfs_2m_dewpoint",
    "sun_elevation",
]

file_unimportant_columns = {
    "circuits.csv": ["url"],
    "constructors.csv": ["url"],
    "drivers.csv": ["url"],
    "races.csv": ["url"],
    "seasons.csv": ["url"],
}

file_important_columns = {
    "circuits.csv": ["circuitId", "name", "location", "country"],
    "constructor_results.csv": [
        "constructorResultsId",
        "raceId",
        "constructorId",
        "points",
    ],
    "constructor_standings.csv": [
        "constructorStandingsId",
        "raceId",
        "constructorId",
        "points",
        "position",
    ],
    "constructors.csv": ["constructorId", "name", "nationality"],
    "lap_times.csv": ["raceId", "driverId", "lap", "milliseconds"],
    "pit_stops.csv": [
        "raceId",
        "driverId",
        "stop",
        "duration",
        "milliseconds",
    ],
    "drivers.csv": ["driverId", "forename", "surname", "dob", "nationality"],
    "driver_standings.csv": [
        "driverStandingsId",
        "raceId",
        "driverId",
        "points",
        "position",
    ],
    "qualifying.csv": ["qualifyId", "raceId", "driverId", "constructorId", "position"],
    "races.csv": ["raceId", "year", "circuitId", "name", "date", "time"],
    "results.csv": [
        "resultId",
        "raceId",
        "driverId",
        "constructorId",
        "points",
        "position",
    ],
    "seasons.csv": ["year"],
    "sprint_results.csv": [
        "resultId",
        "raceId",
        "driverId",
        "constructorId",
        "position",
        "points",
    ],
    "status.csv": ["statusId", "status"],
}


# --------------------------------------- appel des fonctions --------------------------------------


# supprimes les colonnes inutiles
# dropColumnsParquet("weather.parquet", columns_to_drop)

# filtre les données meteo en fonction des courses ayant eu lieu avec les mêmes dates
# filter_by_date("weather.parquet", "races.csv")


# clean_missing_data("weather.parquet", important_columns)
# clean_missing_data("filtered_weather_by_date.parquet", important_columns)


# verification des modifications
# print(readParquet('weather.parquet'))
# list_columns('filtered_weather_by_date.parquet')
# list_columns("weather.parquet")
# list_csv_columns(file_list)


# for csv_file, unimportant_columns in file_unimportant_columns.items():
#     dropColumnsCSV(csv_file, unimportant_columns)

# for csv_file, important_columns in file_important_columns.items():
#     clean_missing_data_csv(csv_file, important_columns)


filter_weather_by_circuit_and_date(
    pd.read_parquet("weather.parquet"),
    pd.read_csv("circuits.csv"),
    pd.read_csv("races.csv"),
    radius_km=20,
    time_window_days=1,
)
