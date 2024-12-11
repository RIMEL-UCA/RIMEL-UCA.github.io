import pandas
import matplotlib.pyplot as plt
import json

def analyze_models(models, models_datas):
    """
    Analyse les données des modèles et affiche des statistiques simples.
    """
    general_data = pandas.DataFrame(models)
    precise_data = pandas.DataFrame(models_datas)

    print("Types de modèles les plus fréquents :")
    print(general_data['pipeline_tag'].value_counts())

    print("\nModèles les plus téléchargés :")
    print(general_data.sort_values(by='downloads', ascending=False)[['id', 'downloads']].head(10))

    print("\nTags les plus fréquents :")
    print(general_data['tags'].apply(pandas.Series).stack().value_counts())

    print("\nModèles les plus anciens :")
    print(general_data.sort_values(by='createdAt', ascending=True))

    return general_data, precise_data

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

    print(pd['_id'].count())

    for i in range(pd['_id'].count()):
        print(f"Index: {i}")
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
            else:
                print(f"L'élément à l'index {i} n'a pas de clé 'total': {safetensor}")
        else:
            print(f"L'élément à l'index {i} n'est pas un dictionnaire : {safetensor}")
    
    print(f"Nombre de modèles SLM : {slm_count}")
    print(f"Nombre de modèles LLM : {llm_count}")
    plt.pie([slm_count, llm_count], labels=['SLM', 'LLM'], autopct='%1.1f%%')
    plt.title("Répartition des modèles SLM et LLM")
    plt.show()