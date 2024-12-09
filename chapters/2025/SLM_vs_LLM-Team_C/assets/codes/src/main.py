from fetch_models import fetch_models, fetch_model_details
from trends import analyze_models, plot_model_types

# Fichiers de données
OUTPUT_FILE = "../data/huggingface_models.json"
MODELS_DATAS_FILE = "../data/models_datas.json"

def main():
    """
    Point d'entrée principal du script.
    """
    try:
        # Récupération des modèles
        models = fetch_models(OUTPUT_FILE)

        # Récupération des détails des modèles
        models_datas = fetch_model_details(models, MODELS_DATAS_FILE)

        # Analyse et visualisation
        gd, dp = analyze_models(models, models_datas)
        plot_model_types(gd)
    except Exception as e:
        print(f"Erreur : {e}")


if __name__ == "__main__":
    main()
