from fetch_models import fetch_models, get_models
from trends import analyze_models, plot_model_types, plot_slm_vs_llm, plot_slm_vs_llm_in_time, when_type_is_most_used, convertJsonToDataFrame, get_parameters_known, plot_most_tags_used, plot_tags_by_time,  plot_monthly_distribution, slm_by_time
import json

# Fichiers de données
# LAST SKIP : 1900
def main():
    """
    Point d'entrée principal du script.
    """
    try:
        # Récupération des modèles
        fetch_models()

        models = get_models()
        analyze_models(models)

        dataframes = convertJsonToDataFrame(models)
        print(f"models: {len(models)}")

        # Génération des graphiques
        plot_tags_by_time("multimodal")
        slm_by_time()
        plot_model_types(dataframes)
        plot_slm_vs_llm(dataframes)
        plot_slm_vs_llm_in_time(dataframes)
        plot_tags_by_time(dataframes)
        #For LLM , what are the most tags used
        llm_most_tags = when_type_is_most_used("llm",dataframes)
        # v > X to filter tags used less than X times
        # filtered_tags = {k: v for k, v in llm_most_tags.items() if v > 10}
        # plot_most_tags_used("llm", filtered_tags)
        plot_monthly_distribution(dataframes)
        filtered_tags = {k: v for k, v in llm_most_tags.items() if v > 10}
        plot_most_tags_used("llm", filtered_tags)
        #For SLM , what are the most tags used
        print(sorted(when_type_is_most_used("slm",dataframes).items(), key=lambda x: x[1], reverse=True))
    except Exception as e:
        print(f"Erreur : {e}")

if __name__ == "__main__":
    main()
