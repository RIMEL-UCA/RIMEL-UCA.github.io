import pandas
import matplotlib.pyplot as plt
import datetime

SLM_LLM_THRESHOLD = 7000000000
NOT_RELEVENT_TAGS = ["en", "fr", "it", "pt", "hi", "es", "th", "de", "ko", "zh", "ja", "pytorch", "llama", "conversational", "autotrain_compatible", "safetensors", "endpoints_compatible","region:us"]
# Words that tags should not contain
NOT_RELEVENT_TAGS_WORDS = ["license", "region", "qwen"]

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
                                if len(tag) > 2 and tag not in NOT_RELEVENT_TAGS:
                                    if tag in type_most_used:
                                        type_most_used[tag] += 1
                                    else:
                                        type_most_used[tag] = 1
                    else:
                        if type == "slm":
                            if safetensor['total'] < 7000000000:
                                tags = pd['tags'][i]
                                for tag in tags:
                                    if len(tag) > 2 and tag not in NOT_RELEVENT_TAGS:
                                        if tag in type_most_used:
                                            type_most_used[tag] += 1
                                        else:
                                            type_most_used[tag] = 1
    return type_most_used
    
def is_slm(model):
    """
    Détermine si un modèle est un modèle SLM.
    """
    safetensors = model.get('safetensors')
    if isinstance(safetensors, dict) and 'total' in safetensors:
        return safetensors.get('total') <= SLM_LLM_THRESHOLD
    return False

def get_creation_date(model):
    """
    Récupère la date de création d'un modèle au format datetime.
    """
    if 'createdAt' in model:
        return datetime.datetime.fromisoformat(model['createdAt'])
    return None

def get_creation_date_from_model(model):
    """
    Récupère la date de création du modèle sous forme de datetime.

    """
    return get_creation_date(model) if 'createdAt' in model else None

def sanitize_tags(tags):
    """
    Filtre et supprime les tags non pertinents des données.
    """
    if not isinstance(tags, list):
        return [];

    return [
        tag for tag in tags
        if tag.lower() not in NOT_RELEVENT_TAGS  # Tags spécifiques à exclure
           and not any(word in tag.lower() for word in NOT_RELEVENT_TAGS_WORDS)  # Mots-clés exclus
    ]

def update_slm_percentage(slm_percentages, tag, month_year, is_slm_model):
    """
    Met à jour le pourcentage de SLM pour un tag donné dans un mois donné.
    """
    if month_year not in slm_percentages[tag]:
        slm_percentages[tag][month_year] = {"slm": 0, "total": 0}

    slm_percentages[tag][month_year]["total"] += 1
    if is_slm_model:
        slm_percentages[tag][month_year]["slm"] += 1

def plot_tags_by_time(models):
    """
    Génère des graphiques pour les n tags les plus populaires avec l'évolution
    du pourcentage de SLM au fil du temps.

    """
    n = 10
    # Vérifier les colonnes nécessaires
    required_columns = {'tags', 'createdAt', 'safetensors'}
    if not required_columns.issubset(models.columns):
        print(f"Erreur : Colonnes nécessaires manquantes. Requis : {required_columns}")
        return

    # Récupérer tous les tags et les nettoyer
    all_tags = models['tags'].apply(sanitize_tags).explode()
    if all_tags.empty:
        print("Aucun tag valide trouvé après nettoyage.")
        return

    # Identifier les n tags les plus populaires
    top_tags = all_tags.value_counts().head(n).index.tolist()
    if not top_tags:
        print("Aucun tag populaire trouvé.")
        return

    # Initialiser un dictionnaire pour stocker les données SLM et totales par mois pour chaque tag
    slm_data = {tag: {} for tag in top_tags}

    # Boucler sur les modèles et collecter les données
    for _, model in models.iterrows():
        tags = model.get('tags')
        if not tags:
            continue

        # Récupérer la date de création et le mois/année
        creation_date = get_creation_date_from_model(model)
        if not creation_date:
            continue
        month_year = creation_date.strftime("%Y-%m")

        # Vérifier si le modèle est SLM
        is_slm_model =  is_slm(model)

        # Mettre à jour les données pour chaque tag
        for tag in tags:
            if tag not in top_tags:
                continue
            slm_data[tag].setdefault(month_year, {"slm": 0, "total": 0})
            slm_data[tag][month_year]["total"] += 1
            if is_slm_model:
                slm_data[tag][month_year]["slm"] += 1

    # Créer des graphiques pour chaque tag
    for tag, data in slm_data.items():
        # Calculer les pourcentages SLM par mois
        percentages = {
            month: (counts["slm"] / counts["total"]) * 100
            for month, counts in data.items()
        }
        slms = {
            month: counts["slm"]
            for month, counts in data.items()
        }

        # Trier les données par mois
        sorted_months = sorted(percentages.keys())
        sorted_percentages = [percentages[month] for month in sorted_months]
        sorted_slms = [slms[month] for month in sorted_months]


        # Générer le graphique
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Axe principal : Pourcentage de SLM
        ax1.plot(sorted_months, sorted_percentages, marker='o', color='b', label="Pourcentage SLM")
        ax1.set_title(f"Tag : {tag} - Pourcentage de SLM et Total par mois", fontsize=14)
        ax1.set_xlabel("Temps (mois/année)", fontsize=12)
        ax1.set_ylabel("Pourcentage de SLM (%)", color='b', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True)

        # Ajuster les étiquettes pour éviter le chevauchement
        ax1.set_xticks(range(len(sorted_months)))  # Positionner les étiquettes
        ax1.set_xticklabels(
            [month if i % 2 == 0 else "" for i, month in enumerate(sorted_months)],  # Un mois sur deux
            rotation=45,
            fontsize=10
        )

        # Axe secondaire : Nombre total d'occurrences
        ax2 = ax1.twinx()
        ax2.bar(sorted_months, sorted_slms, alpha=0.3, color='gray', label="Total", width=0.4)
        ax2.set_ylabel("Nombre de SLM", color='gray', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='gray')

        # Ajuster l'affichage
        fig.tight_layout()
        plt.xticks(rotation=45)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.show()