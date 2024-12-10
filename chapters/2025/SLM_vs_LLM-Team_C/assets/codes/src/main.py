from fetch_models import fetch_models, fetch_model_details
from trends import analyze_models, plot_model_types, plot_slm_vs_llm
import json

# Fichiers de données

def main():
    """
    Point d'entrée principal du script.
    """
    try:
        # Récupération des modèles
        models, models_datas = fetch_models()
        gd, pd = analyze_models(models, models_datas)

        plot_slm_vs_llm(pd)
    except Exception as e:
        print(f"Erreur : {e}")

if __name__ == "__main__":
    main()
