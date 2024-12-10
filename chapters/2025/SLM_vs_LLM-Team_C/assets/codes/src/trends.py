import pandas as pd
import matplotlib.pyplot as plt

def analyze_models(models, models_datas):
    """
    Analyse les données des modèles et affiche des statistiques simples.
    """
    gd = pd.DataFrame(models)
    dp = pd.DataFrame(models_datas)

    print("Types de modèles les plus fréquents :")
    print(gd['pipeline_tag'].value_counts())

    print("\nModèles les plus téléchargés :")
    print(gd.sort_values(by='downloads', ascending=False)[['id', 'downloads']].head(10))

    print("\nTags les plus fréquents :")
    print(gd['tags'].apply(pd.Series).stack().value_counts())

    print("\nModèles les plus anciens :")
    print(gd.sort_values(by='createdAt', ascending=True))

    return gd, dp

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
