from fetch_models import fetch_models, fetch_model_details
from trends import analyze_models, plot_model_types, plot_slm_vs_llm, plot_slm_vs_llm_in_time, when_type_is_most_used, convertJsonToDataFrame, get_parameters_known, plot_most_tags_used
import json

# Fichiers de données
# LAST SKIP : 1400
def main():
    """
    Point d'entrée principal du script.
    """
    try:
        # Récupération des modèles
        models, models_datas = fetch_models()

        #gd, pd = analyze_models(models, models_datas)

        pd = convertJsonToDataFrame(models_datas)
        print("parameters known : ", get_parameters_known(pd))

        # Génération des graphiques
        #plot_model_types(gd)
        #plot_slm_vs_llm(pd)
        #plot_slm_vs_llm_in_time(pd)
        #For LLM , what are the most tags used
        llm_most_tags = when_type_is_most_used("llm",pd)
        # v > X to filter tags used less than X times
        filtered_tags = {k: v for k, v in llm_most_tags.items() if v > 10}
        plot_most_tags_used("llm", filtered_tags)
        #For SLM , what are the most tags used
        #print(sorted(when_type_is_most_used("slm",pd).items(), key=lambda x: x[1], reverse=True))
    except Exception as e:
        print(f"Erreur : {e}")

if __name__ == "__main__":
    main()
