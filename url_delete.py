import os
import pandas as pd

def remove_url_column_from_csv(file_path):
    try:
        print(f"Traitement du fichier : {file_path}")
        # Lire le fichier CSV
        df = pd.read_csv(file_path)
        
        # Vérifier si la colonne 'url' existe et la supprimer
        if 'url' in df.columns:
            df.drop(columns=['url'], inplace=True)
            # Sauvegarder le fichier CSV modifié
            df.to_csv(file_path, index=False)
            print(f"Colonne 'url' supprimée de {file_path}")
        else:
            print(f"Aucune colonne 'url' trouvée dans {file_path}")
    except Exception as e:
        print(f"Erreur lors du traitement de {file_path}: {e}")

def process_files(files):
    for file in files:
        if os.path.isfile(file):
            remove_url_column_from_csv(file)
        else:
            print(f"Fichier non trouvé : {file}")

if __name__ == "__main__":
    # Liste des fichiers à traiter
    files_to_process = [
        'circuits.csv',
        'constructor_results.csv',
        'constructor_standings.csv',
        'constructors.csv',
        'drivers.csv',
        'driver_standings.csv',
        'lap_times.csv',
        'pit_stops.csv',
        'qualifying.csv',
        'races.csv',
        'results.csv',
        'seasons.csv',
        'status.csv'
    ]
    
    print("Début du traitement des fichiers")
    process_files(files_to_process)
    print("Fin du traitement des fichiers")