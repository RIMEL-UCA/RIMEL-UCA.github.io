
import pandas
import matplotlib.pyplot as plt
import datetime
from fetch_models  import get_models

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
    print(models)
    """
    Analyse les données des modèles et affiche des statistiques simples.
    """
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




def update_slm_percentage(slm_percentages, perio, is_slm_model):
    """
    Met à jour le pourcentage de SLM pour un tag donné dans un mois donné.
    """
    if perio not in slm_percentages:
        slm_percentages[perio] = {"slm": 0, "total": 0}

    slm_percentages[perio]["total"] += 1
    if is_slm_model:
        slm_percentages[perio]["slm"] += 1



def slm_by_time():
    """
    Génère des graphiques pour les n tags les plus populaires avec l'évolution
    du pourcentage de SLM au fil du temps.

    """
    models = get_models()
    dataframes = convertJsonToDataFrame(models)

    slm_data = {}

    # Boucler sur les modèles et collecter les données
    for _, model in dataframes.iterrows():

        # Récupérer la date de création et le mois/année
        creation_date = get_creation_date_from_model(model)
        if not creation_date:
            continue

        if creation_date.year < 2023:
            continue

        month_year = creation_date.strftime("%Y-%m")
        if month_year == datetime.datetime.now().strftime("%Y-%m"):
            continue


        # Vérifier si le modèle est SLM
        is_slm_model =  is_slm(model)

        update_slm_percentage(slm_data, month_year, is_slm_model)

    periods = sorted(slm_data.keys())
    percentages = [
        ((slm_data[period]["slm"] / slm_data[period]["total"]) * 100) if slm_data[period][
                                                                                     "total"] > 0 else 0
        for period in periods
    ]

    # Diagramme en bâtons
    plt.figure(figsize=(10, 5))
    plt.bar(periods, percentages, color='skyblue')
    plt.title(f"Pourcentage de SLM créés par mois", fontsize=14)
    plt.xlabel("Mois", fontsize=12)
    plt.ylabel("Pourcentage de SLM (%)", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.show()

def plot_tags_by_time(tag):
    """
    Génère des graphiques pour les n tags les plus populaires avec l'évolution
    du pourcentage de SLM au fil du temps.

    """
    models = get_models(tag)
    dataframes = convertJsonToDataFrame(models)

    slm_data = {}

    # Boucler sur les modèles et collecter les données
    for _, model in dataframes.iterrows():

        # Récupérer la date de création et le mois/année
        creation_date = get_creation_date_from_model(model)
        if not creation_date:
            continue

        if creation_date.year < 2023:
            continue

        month_year = creation_date.strftime("%Y-%m")
        if month_year == datetime.datetime.now().strftime("%Y-%m"):
            continue


        # Vérifier si le modèle est SLM
        is_slm_model =  is_slm(model)

        update_slm_percentage(slm_data, month_year, is_slm_model)

    periods = sorted(slm_data.keys())
    percentages = [
        ((slm_data[period]["slm"] / slm_data[period]["total"]) * 100) if slm_data[period][
                                                                                     "total"] > 0 else 0
        for period in periods
    ]

    # Diagramme en bâtons
    plt.figure(figsize=(10, 5))
    plt.bar(periods, percentages, color='skyblue')
    plt.title(f"Pourcentage de SLM créés par mois pour le tag '{tag}'", fontsize=14)
    plt.xlabel("Année/Mois de création", fontsize=12)
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



    # Calculer les pourcentages pour chaque mois
    slm_percentages =  [
        (slm_data.get(month, 0) / slm_total) * 100 if slm_total > 0 else 0
        for month in all_months
    ]
    llm_percentages = [
        (llm_data.get(month, 0) / llm_total) * 100 if llm_total > 0 else 0
        for month in all_months
    ]

    # Plot 1: SLM
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(all_months)), slm_percentages, color="skyblue", width=0.5)
    plt.title("SLM - Part des téléchargements des 30 derniers jours en fonction de la date de création", fontsize=14)
    plt.xlabel("Mois/Année de création", fontsize=12)
    plt.ylabel("Part des téléchargements (%)", fontsize=12)
    plt.xticks(range(len(all_months)), all_months, rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plot 2: LLM
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(all_months)), llm_percentages, color="orange", width=0.5)
    plt.title("LLM - Part des téléchargements des 30 derniers jours en fonction de la date de création", fontsize=14)
    plt.xlabel("Mois/Année de création", fontsize=12)
    plt.ylabel("Part des téléchargements (%)", fontsize=12)
    plt.xticks(range(len(all_months)), all_months, rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


