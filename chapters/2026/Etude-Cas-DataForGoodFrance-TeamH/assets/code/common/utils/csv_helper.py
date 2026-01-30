import pandas as pd
from pathlib import Path
import sys

class CsvHelper:
    @staticmethod
    def read_and_validate(file_path, required_columns=None):
        """
        Lit un CSV de manière robuste (sep , ou ;), normalise les colonnes
        et vérifie la présence des champs requis.
        
        :param file_path: Chemin vers le fichier (str ou Path)
        :param required_columns: Liste des colonnes attendues (ex: ['email', 'job'])
        :return: DataFrame nettoyé ou None si erreur
        """
        path = Path(file_path)
        if not path.exists():
            print(f"Fichier introuvable : {path}")
            return None

        try:
            try:
                df = pd.read_csv(path, encoding='utf-8')
                if len(df.columns) <= 1: 
                    df = pd.read_csv(path, sep=';', encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(path, sep=None, engine='python', encoding='latin1')

            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

            if required_columns:
                req_norm = [c.lower().strip() for c in required_columns]
                missing = [c for c in req_norm if c not in df.columns]
                
                if missing:
                    print(f"Colonnes manquantes dans {path.name} : {missing}")
                    print(f"Colonnes trouvées : {list(df.columns)}")
                    return None

            df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

            print(f"Chargement réussi : {path.name} ({len(df)} lignes)")
            return df

        except Exception as e:
            print(f"Erreur critique lors de la lecture de {path.name}: {e}")
            return None

    @staticmethod
    def clean_cell_value(val, default_value='Unknown'):
        """Nettoie une valeur unitaire (pour un job ou un nom)"""
        val = str(val).strip()
        if val.lower() in ['nan', 'none', '', '?', 'null', 'unknown']:
            return default_value
        return val.replace('?', '').strip()