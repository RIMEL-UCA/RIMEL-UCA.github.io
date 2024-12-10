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

    pd['safetensors'].values.tolist()[1]['total']

    llm_count = 0
    slm_count = 0
    if 'safetensors' in pd.columns:
        for obj in pd['safetensors'].values:
            #//TODO: Check if the key 'total' exists in the object
            print()
    else:
        print("La colonne 'safetensors' est absente du DataFrame.")
    print(slm_count, llm_count)
