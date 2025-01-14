import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

# Constantes
API_URL = "https://huggingface.co/api/models"
HUGGINFACE_KEY = os.getenv("HUGGINFACE_KEY")  # Charger la clé depuis .env
HEADERS = {"Authorization": f"Bearer {HUGGINFACE_KEY}"}

MODELS_DATAS_FILE = "../data/models_datas.json"

def get_models(tag=""):
    """
    Retourne les modèles depuis un fichier local.
    """
    models = []
    file_name = MODELS_DATAS_FILE
    if tag:
        file_name = f"../data/{tag}_models_datas.json"

    # if the file exists
    if os.path.exists(file_name):
        return load_json_from_file(file_name)
    else :
        return []

def load_json_from_file(file_path):
    """
    Charge un fichier JSON et retourne son contenu.
    """
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    return None


def save_json_to_file(file_path, data):
    """
    Sauvegarde les données au format JSON dans un fichier.
    """
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def fetch_from_api(params, path_complement=""):
    """
    Effectue une requête à l'API HuggingFace et retourne la réponse JSON si elle est valide.
    """
    response = requests.get(API_URL + path_complement, params=params, headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Erreur lors de la requête API : {response.status_code}")


def fetch_models():
    """
    Récupère les modèles depuis l'API ou un fichier local.
    """
    tag_choice = input("Voulez-vous récupérer des données pour un tag spécifique ? (o/n) : ")
    tag = ""
    if tag_choice.lower() == "o":
        tag = input("Entrez le tag spécifique pour lequel vous souhaitez les données : ")
    number_models = int(input("Entrez le nombre de modèles à récupérer : "))
    limit = 1000
    for i in range(number_models // limit):
        print(f"Récupération des modèles {i * limit} à {i * limit + limit}.")
        models = get_models(tag)
        new_models = []
        PARAMS = {"limit": limit, "skip": i * limit, "sort": "downloads"}
        if tag:
            PARAMS["filter"] = tag
        new_models = fetch_from_api(PARAMS)

        # Retirer les tous modèles déjà present de new_model
        new_models = [model for model in new_models if not any(m.get("_id") == model.get("_id") for m in models)]

        fetch_model_details(new_models, tag)



def fetch_model_details(models, tag=""):
    """
    Récupère les détails des modèles depuis l'API ou un fichier local.
    """
    new_models_details = []
    for model in models:
        # print(f"Récupération des détails pour le modèle {model['id']}.")

        params = {"full": "True"}
        model_data = fetch_from_api(params, f"/{model['id']}")

        if "safetensors" in model_data and "parameters" in model_data["safetensors"]:
            new_models_details.append(model_data)

    file_name = MODELS_DATAS_FILE
    if tag:
        file_name = f"../data/{tag}_models_datas.json"

    rsl = get_models()
    rsl.extend(new_models_details)
    save_json_to_file(file_name, rsl)

    print(f"Données détaillées sauvegardées dans le fichier '{file_name}'.")