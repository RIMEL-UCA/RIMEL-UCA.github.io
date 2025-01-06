import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
# Constantes
API_URL = "https://huggingface.co/api/models"
HUGGINFACE_KEY = os.getenv("HUGGINFACE_KEY")  # Charger la clé depuis .env
HEADERS = {"Authorization": f"Bearer {HUGGINFACE_KEY}"}

OUTPUT_FILE = "../data/huggingface_models.json"
MODELS_DATAS_FILE = "../data/models_datas.json"

def fetch_models():
    """
    Récupère les modèles depuis l'API ou un fichier local.
    """
    models = None
    if os.path.exists(OUTPUT_FILE):
        #regarder le nombre de modèles dans le fichier
        with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
            print("Chargement des modèles depuis le fichier local.")
            #compute json object
            models = json.load(file)
            print(f"{len(models)} modèles chargés.")
            input_user = input("Voulez-vous récupérer les dernières données ? (o/n) : ")
            if input_user.lower() == "o":
                input_user = input("On skip combien de modèles (plus tu en enlèves, plus on va aller chercher loin dans les modeles en ligne) ? : ")
                if input_user.isdigit():
                    PARAMS = {"limit": 100, "skip": int(input_user), "sort": "downloads"}
                    response = requests.get(API_URL, headers=HEADERS, params=PARAMS)
                    if response.status_code == 200:
                        new_models = response.json()
                        for model in models:
                            for m in new_models:
                            #on retire les modèles déjà présents (si jamais on pull plusieurs fois les mêmes modèles)
                                if model.get("_id") == m.get("_id"):
                                    print(f"Le modèle {model.get('_id')} est déjà présent.")
                                    new_models.remove(m)
                        models.extend(new_models)
                        with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
                            json.dump(models, file, ensure_ascii=False, indent=4)
                        print(f"{len(new_models)} nouveaux modèles ajoutés.")
            else:
                print("Données chargées depuis le fichier local. Sans ajout de modèles.")
    else:
        PARAMS = {"limit": 100, "skip": 0}
        response = requests.get(API_URL, headers=HEADERS, params=PARAMS)
        if response.status_code == 200:
            models = response.json()
            with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
                json.dump(models, file, ensure_ascii=False, indent=4)
            print(f"Données sauvegardées dans le fichier '{OUTPUT_FILE}'.")
        else:
            raise Exception(f"Erreur lors de la requête API : {response.status_code}")

    models_details = fetch_model_details(models)
    return models, models_details


def fetch_model_details(models):
    """
    Récupère les détails des modèles depuis l'API ou un fichier local.
    """
    if os.path.exists(MODELS_DATAS_FILE):
        with open(MODELS_DATAS_FILE, "r", encoding="utf-8") as file:
            print("Chargement des détails des modèles depuis le fichier local.")
            models_datas =  json.load(file)
            if len(models_datas) >= len(models):
                return models_datas
            else:
                print(f"Le nombre de modèles dans le fichier ({len(models_datas)}) ne correspond pas au nombre de modèles récupérés ({len(models)}).")
                for model in models:
                    if not any(model.get("_id") == m.get("_id") for m in models_datas):
                        print(f"Le modèle {model.get('id')} n'est pas présent dans le fichier.")
                        response = requests.get(
                            f"https://huggingface.co/api/models/{model.get('id')}",
                            params={"full": "True"},
                            headers=HEADERS,
                        )
                        if response.status_code == 200:
                            models_datas.append(response.json())
                        else:
                            print(f"Erreur pour le modèle {model.get('id')}: {response.status_code}")
                with open(MODELS_DATAS_FILE, "w", encoding="utf-8") as file:
                    json.dump(models_datas, file, ensure_ascii=False, indent=4)
                print(f"Données détaillées sauvegardées dans le fichier '{MODELS_DATAS_FILE}'.")
    else:
        print("Aucun fichier, \nChargement des détails des modèles depuis l'API.")
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
            print(f"{model['id']} modèles détaillés récupérés.")
        with open(MODELS_DATAS_FILE, "w", encoding="utf-8") as file:
            json.dump(models_datas, file, ensure_ascii=False, indent=4)
        print(f"Données détaillées sauvegardées dans le fichier '{MODELS_DATAS_FILE}'.")
    
    return models_datas
