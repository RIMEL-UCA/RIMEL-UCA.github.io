import os
import json
import requests
from dotenv import load_dotenv

load_dotenv();
# Constantes
API_URL = "https://huggingface.co/api/models"
HUGGINFACE_KEY = os.getenv("HUGGINFACE_KEY")  # Charger la clé depuis .env
HEADERS = {"Authorization": f"Bearer {HUGGINFACE_KEY}"}
PARAMS = {"limit": 50}

def fetch_models(output_file):
    """
    Récupère les modèles depuis l'API ou un fichier local.
    """
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as file:
            print("Chargement des modèles depuis le fichier local.")
            return json.load(file)
    else:
        response = requests.get(API_URL, headers=HEADERS, params=PARAMS)
        if response.status_code == 200:
            models = response.json()
            with open(output_file, "w", encoding="utf-8") as file:
                json.dump(models, file, ensure_ascii=False, indent=4)
            print(f"Données sauvegardées dans le fichier '{output_file}'.")
            return models
        else:
            raise Exception(f"Erreur lors de la requête API : {response.status_code}")


def fetch_model_details(models, output_file):
    """
    Récupère les détails des modèles depuis l'API ou un fichier local.
    """
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as file:
            print("Chargement des détails des modèles depuis le fichier local.")
            return json.load(file)
    else:
        models_datas = []
        for model in models:
            response = requests.get(
                f"https://huggingface.co/api/models/{model.get('id')}",
                params={"full": "True"},
                headers=HEADERS,
            )
            if response.status_code == 200:
                models_datas.append(response.json())
            else:
                print(f"Erreur pour le modèle {model.get('id')}: {response.status_code}")
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(models_datas, file, ensure_ascii=False, indent=4)
        print(f"Données détaillées sauvegardées dans le fichier '{output_file}'.")
        return models_datas
