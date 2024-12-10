from fetch_models import fetch_models, fetch_model_details
from trends import analyze_models, plot_model_types, plot_slm_vs_llm

# Fichiers de données

def main():
    """
    Point d'entrée principal du script.
    """
    try:
        # Récupération des modèles
        models, models_datas = fetch_models()
        gd, dp = analyze_models(models, models_datas)
        plot_model_types(gd)
        plot_slm_vs_llm(dp)
    except Exception as e:
        print(f"Erreur : {e}")

if __name__ == "__main__":
    main()
