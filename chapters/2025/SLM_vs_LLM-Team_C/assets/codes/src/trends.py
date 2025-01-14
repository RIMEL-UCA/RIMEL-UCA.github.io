
import pandas
import matplotlib.pyplot as plt
import datetime

from fontTools.varLib.mutator import percents

SLM_LLM_THRESHOLD = 7000000000
NOT_RELEVENT_TAGS = ["en", "fr", "it", "pt", "hi", "es", "th", "de", "ko", "zh", "ja", "pytorch", "llama", "conversational", "autotrain_compatible", "safetensors", "endpoints_compatible","region:us"]
# Words that tags should not contain
NOT_RELEVENT_TAGS_WORDS = ["license", "region", "qwen"]

RELEVANT_TAGS = {
    "multimodal", "audio-text-to-text", "image-text-to-text", "visual-question-answering",
    "document-question-answering", "video-text-to-text", "any-to-any", "computer-vision",
    "depth-estimation", "image-classification", "object-detection", "image-segmentation",
    "text-to-image", "image-to-text", "image-to-image", "image-to-video",
    "unconditional-image-generation", "video-classification", "text-to-video",
    "zero-shot-image-classification", "mask-generation", "zero-shot-object-detection",
    "text-to-3d", "image-to-3d", "image-feature-extraction", "keypoint-detection",
    "natural-language-processing", "text-classification", "token-classification",
    "table-question-answering", "question-answering", "zero-shot-classification",
    "translation", "summarization", "feature-extraction", "text-generation",
    "text2text-generation", "fill-mask", "sentence-similarity", "audio",
    "text-to-speech", "text-to-audio", "automatic-speech-recognition", "audio-to-audio",
    "audio-classification", "voice-activity-detection", "tabular",
    "tabular-classification", "tabular-regression", "time-series-forecasting",
    "reinforcement-learning", "robotics", "other", "graph-machine-learning"
}

def convertJsonToDataFrame(json_data):
    return pandas.DataFrame(json_data)

def analyze_models(models):
    """
    Analyse les données des modèles et affiche des statistiques simples.
    """

    general_data = convertJsonToDataFrame(models)
    advanced_data = convertJsonToDataFrame(models)

    print("Types de modèles les plus fréquents :")
    print(advanced_data['pipeline_tag'].value_counts())

    print("\nModèles les plus téléchargés :")
    print(advanced_data.sort_values(by='downloads', ascending=False)[['id', 'downloads']].head(10))

    print("\nTags les plus fréquents :")
    all_tags = advanced_data['tags'].apply(sanitize_tags).explode()
    print(all_tags.value_counts())

    print("\nModèles les plus anciens :")
    print(advanced_data.sort_values(by='createdAt', ascending=True))



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

    plt.pie([slm_count, llm_count], labels=['SLM ('+str(slm_count)+')', 'LLM('+str(llm_count)+')'], autopct='%1.1f%%')
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
    plt.figure(figsize=(15, 6))
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
                    if type == "llm":
                        if safetensor['total'] >= 7000000000:
                            tags = pd['tags'][i]
                            for tag in tags:
                                if tag in RELEVANT_TAGS:
                                    if tag in type_most_used:
                                        type_most_used[tag] += 1
                                    else:
                                        type_most_used[tag] = 1
                    else:
                        if type == "slm":
                            if safetensor['total'] < 7000000000:
                                tags = pd['tags'][i]
                                for tag in tags:
                                    if tag in RELEVANT_TAGS:
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
    Filtre les tags pour conserver uniquement ceux correspondant aux catégories spécifiées.
    """
    if not isinstance(tags, list):
        return []

    # Keep only tags that match the relevant categories
    return [tag for tag in tags if tag in RELEVANT_TAGS]




def update_slm_percentage(slm_percentages, tag, perio, is_slm_model):
    """
    Met à jour le pourcentage de SLM pour un tag donné dans un mois donné.
    """
    if perio not in slm_percentages[tag]:
        slm_percentages[tag][perio] = {"slm": 0, "total": 0}

    slm_percentages[tag][perio]["total"] += 1
    if is_slm_model:
        slm_percentages[tag][perio]["slm"] += 1


def plot_tags_by_time(models):
    """
    Génère des graphiques pour les n tags les plus populaires avec l'évolution
    du pourcentage de SLM au fil du temps.

    """
    n = 6
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

    print("Tags valides :")
    print(all_tags.value_counts())

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

        if creation_date.year < 2023:
            continue

        month_year = creation_date.strftime("%Y-%m")
        if month_year == datetime.datetime.now().strftime("%Y-%m"):
            continue

        # Calculer la période (groupe de 4 mois)
        month = creation_date.month
        start_month = (month - 1) // 4 * 4 + 1
        period_start = datetime.date(creation_date.year, start_month, 1)
        period_end = datetime.date(creation_date.year, start_month + 3, 1)
        period = f"{period_start.strftime('%Y-%m')} à {period_end.strftime('%Y-%m')}"

        # Vérifier si le modèle est SLM
        is_slm_model =  is_slm(model)

        # Mettre à jour les données pour chaque tag
        for tag in tags:
            if tag in top_tags:
                update_slm_percentage(slm_data, tag, period, is_slm_model)

    for tag in top_tags:
        periods = sorted(slm_data[tag].keys())
        percentages = [
            (slm_data[tag][period]["slm"] / slm_data[tag][period]["total"] * 100) if slm_data[tag][period][
                                                                                         "total"] > 0 else 0
            for period in periods
        ]

        # Diagramme en bâtons
        plt.figure(figsize=(10, 5))
        plt.bar(periods, percentages, color='skyblue')
        plt.title(f"Évolution du pourcentage de SLM pour le tag '{tag}' (par période de 4 mois)", fontsize=14)
        plt.xlabel("Période", fontsize=12)
        plt.ylabel("Pourcentage de SLM (%)", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        plt.show()





def group_months_by_period(months, period_length=4):
    # Grouper les mois en périodes de 4 mois
    periods = []
    current_period = []
    for month in months:
        current_period.append(month)
        if len(current_period) == period_length:
            periods.append(f"{current_period[0]} à {current_period[-1]}")
            current_period = []
    # Ajouter la période restante si elle n'est pas vide
    if current_period:
        periods.append(f"{current_period[0]} à {current_period[-1]}")
    return periods

def plot_monthly_distribution(models):
    # Initialiser un dictionnaire pour stocker les donwloads SLM et SLM et totales par mois
    slm_data = {}
    slm_total = 0
    llm_data = {}
    llm_total = 0

    # Boucler sur les modèles et collecter les données
    for _, model in models.iterrows():
        # Si le modèle n'a pas de nombre de safe tensors, passer au suivant
        if not model.get('safetensors'):
            continue


        # Récupérer la date de création et le mois/année
        creation_date = get_creation_date_from_model(model)
        if not creation_date:
            continue

        # si le modèle date d'avant 2023 continue
        if creation_date.year < 2023:
            continue

        # si le modèle date de notre mois curre
        now = datetime.datetime.now().strftime("%Y-%m")
        if creation_date.strftime("%Y-%m") == now:
            continue
        month_year = creation_date.strftime("%Y-%m")

        # Vérifier si le modèle est SLM
        is_slm_model = is_slm(model)
        num_downloads = model.get('downloads', 0)

        # Mettre à jour les données pour chaque type
        if is_slm_model:
            slm_data[month_year] = slm_data.get(month_year, 0) + num_downloads
            slm_total += num_downloads
        else:
            llm_data[month_year] = llm_data.get(month_year, 0) + num_downloads
            llm_total += num_downloads

    all_months = sorted(set(list(slm_data.keys()) + list(llm_data.keys())))
    cumulated_slm_data = {month: sum(slm_data.get(m, 0) for m in all_months[:i+1]) for i, month in enumerate(all_months)}
    cumulated_llm_data = {month: sum(llm_data.get(m, 0) for m in all_months[:i+1]) for i, month in enumerate(all_months)}



    # Calculer les pourcentages pour chaque mois
    slm_percentages = [
        (cumulated_slm_data.get(month, 0) / slm_total) * 100 if slm_total > 0 else 0
        for month in all_months
    ]
    llm_percentages = [
        (cumulated_llm_data.get(month, 0) / llm_total) * 100 if llm_total > 0 else 0
        for month in all_months
    ]

    # Création du graphique
    fig, ax = plt.subplots(figsize=(12, 6))

    # Tracer les barres pour SLM et LLM
    bar_width = 0.35
    x = range(len(all_months))
    ax.bar(x, slm_percentages, width=bar_width, label="SLM", color="skyblue")
    ax.bar([i + bar_width for i in x], llm_percentages, width=bar_width, label="LLM", color="orange")

    # Ajouter des annotations et une légende
    ax.set_title("Distribution des téléchargments du mois dernier en fonction de la date de cré", fontsize=14)
    ax.set_xlabel("Mois/Année de Création", fontsize=12)
    ax.set_ylabel("Part des téléchargements du mois dernier", fontsize=12)
    ax.set_xticks([i + bar_width / 2 for i in x])
    ax.set_xticklabels(all_months, rotation=45)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


