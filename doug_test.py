import pandas as pd


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


# print(check_locations_in_weather("circuits.csv", "daily_weather.parquet"))


def check_cities_in_csv(cities_list, csv_file, column_name):
    data = pd.read_csv(csv_file)
    csv_cities = data[column_name].unique()
    present_cities = []
    missing_cities = []
    for city in cities_list:
        if city in csv_cities:
            present_cities.append(city)
        else:
            missing_cities.append(city)
    print(f"Villes présentes dans {csv_file} : {present_cities}")
    print(f"Villes manquantes dans {csv_file} : {missing_cities}")

    return present_cities, missing_cities


cities_to_check = check_locations_in_weather("circuits.csv", "daily_weather.parquet")

present_cities_cities_csv, missing_cities_cities_csv = check_cities_in_csv(
    cities_to_check, "cities.csv", "city_name"
)

# je vois pas dans quel monde y a mais hassoul ça coute rien de tester
present_cities_countries_csv, missing_cities_countries_csv = check_cities_in_csv(
    cities_to_check, "countries.csv", "native_name"
)