import pandas
import matplotlib.pyplot as plt
import datetime

def convertJsonToDataFrame(json_data):
    return pandas.DataFrame(json_data)

def analyze_models(models, models_datas):
    """
    Analyse les données des modèles et affiche des statistiques simples.
    """

    general_data = convertJsonToDataFrame(models)
    advanced_data = convertJsonToDataFrame(models_datas)

    print("Types de modèles les plus fréquents :")
    print(general_data['pipeline_tag'].value_counts())

    print("\nModèles les plus téléchargés :")
    print(general_data.sort_values(by='downloads', ascending=False)[['id', 'downloads']].head(10))

    print("\nTags les plus fréquents :")
    print(general_data['tags'].apply(pandas.Series).stack().value_counts())

    print("\nModèles les plus anciens :")
    print(general_data.sort_values(by='createdAt', ascending=True))

    return general_data, advanced_data

def get_parameters_known(pd):
    """
    Récupère les paramètres connus.
    """
    toret = 0
    for i in range(pd['_id'].count()):
        # Récupère l'élément
        safetensor = pd['safetensors'].values.tolist()[i]
        # Vérifie que l'élément est un dictionnaire
        if isinstance(safetensor, dict):
            if 'total' in safetensor:
                toret += 1
    return toret

def plot_model_types(df):
    """
    Génère un graphique de distribution des types de modèles.
    """
    if 'pipeline_tag' in df.columns:
        type_counts = df['pipeline_tag'].value_counts()
        type_counts.plot(kind='bar', figsize=(10, 6), title="Distribution des types de modèles")
        plt.xlabel("Type de modèle")
        plt.ylabel("Nombre de modèles")
        plt.show()
    else:
        print("La colonne 'pipeline_tag' est absente du DataFrame.")

def plot_slm_vs_llm(pd):
    """
    Génère un graphique de comparaison entre les modèles SLM et LLM.
    """
    print("PLOT SLM VS LLM")

    print(pd['safetensors'].values.tolist()[2]['total'])

    llm_count = 0
    slm_count = 0

    for i in range(pd['_id'].count()):
        # Récupère l'élément
        safetensor = pd['safetensors'].values.tolist()[i]
        
        # Vérifie que l'élément est un dictionnaire
        if isinstance(safetensor, dict):
            # Vérifie que la clé 'total' existe
            if 'total' in safetensor:
                if safetensor['total'] >= 7000000000:
                    llm_count += 1
                else:
                    slm_count += 1
    
    plt.pie([slm_count, llm_count], labels=['SLM', 'LLM'], autopct='%1.1f%%')
    plt.title("Répartition des modèles SLM et LLM")
    plt.show()

def plot_slm_vs_llm_in_time(pd):
    """
    Génère un graphique de comparaison entre les modèles SLM et LLM dans le temps.
    """
    print("PLOT SLM VS LLM IN TIME")

    slm_llm_between_months = {}
    total_models_with_safetensors = 0
    for i in range(pd['_id'].count()):
        # Récupère l'élément
        safetensor = pd['safetensors'].values.tolist()[i]
        
        # Vérifie que l'élément est un dictionnaire
        if isinstance(safetensor, dict):
            # Vérifie que la clé 'total' existe
            if 'total' in safetensor:
                # Récupère la date de création du modèle
                date_obj = datetime.datetime.fromisoformat(pd['createdAt'][i])
                # Extraire mois et année
                month_year_key = date_obj.strftime("%Y-%m")
                if safetensor['total'] >= 7000000000:
                    if month_year_key in slm_llm_between_months:
                        slm_llm_between_months[month_year_key]["llm"] += 1
                    else:
                        slm_llm_between_months[month_year_key] = {"llm": 1, "slm": 0}
                else:
                    if month_year_key in slm_llm_between_months:
                       slm_llm_between_months[month_year_key]["slm"] += 1
                    else:
                        slm_llm_between_months[month_year_key] = {"llm": 0, "slm": 1}
                total_models_with_safetensors += 1
    print(f"Nombre total de modèles avec des safetensors : {total_models_with_safetensors}")
    # Trier le dictionnaire par ordre croissant de date
    slm_llm_between_months = {k: v for k, v in sorted(slm_llm_between_months.items(), key=lambda item: item[0])}
    plt.figure(figsize=(24, 12))
    plt.plot(slm_llm_between_months.keys(), [x["slm"] for x in slm_llm_between_months.values()], label='SLM')
    plt.plot(slm_llm_between_months.keys(), [x["llm"] for x in slm_llm_between_months.values()], label='LLM')
    plt.xticks(rotation=45)
    plt.legend()
    plt.title("Répartition des modèles SLM et LLM dans le temps")
    plt.show()

def plot_most_tags_used(model_type, tags_most_used):
    """
    Génère un graphique des tags les plus utilisés pour un type de modèle donné.
    """
    print(f"Tags les plus utilisés pour les modèles {model_type} :")
    print(tags_most_used)
    plt.figure(figsize=(10, 6))
    plt.barh(tags_most_used.keys(), tags_most_used.values())
    plt.xlabel("Tag")
    plt.ylabel("Nombre de modèles")
    plt.title(f"Tags les plus utilisés pour les modèles {model_type}")
    plt.show()

def when_type_is_most_used(type,pd):
    type_most_used = {}
    for i in range(pd['_id'].count()):
            # Récupère l'élément
            safetensor = pd['safetensors'].values.tolist()[i]
            # Vérifie que l'élément est un dictionnaire
            if isinstance(safetensor, dict):
                # Vérifie que la clé 'total' existe
                if 'total' in safetensor:
                    # Extraire mois et année
                    if type == "llm":
                        if safetensor['total'] >= 7000000000:
                            tags = pd['tags'][i]
                            for tag in tags:
                                # Vérifie que le tag a une longueur minimale, evite les tag pour les langues (FR, EN, ...)
                                if len(tag) > 2:
                                    if tag in type_most_used:
                                        type_most_used[tag] += 1
                                    else:
                                        type_most_used[tag] = 1
                    else:
                        if type == "slm":
                            if safetensor['total'] < 7000000000:
                                tags = pd['tags'][i]
                                for tag in tags:
                                    if len(tag) > 2:
                                        if tag in type_most_used:
                                            type_most_used[tag] += 1
                                        else:
                                            type_most_used[tag] = 1
    return type_most_used
    