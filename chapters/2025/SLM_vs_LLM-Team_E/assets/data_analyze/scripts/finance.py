import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# Charger les données du fichier CSV et créer un nuage de points
def plot_cost_vs_params(input_csv):
    # Lire le fichier CSV d'entrée
    df = pd.read_csv(input_csv)

    # S'assurer que les colonnes nécessaires existent
    required_columns = ["Nombre de paramètres (B)", "Coût ($) pour 1K token"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"La colonne '{col}' est manquante dans le fichier CSV d'entrée.")

    # Convertir les colonnes en valeurs numériques (au cas où elles seraient au format texte)
    df["Nombre de paramètres (B)"] = pd.to_numeric(df["Nombre de paramètres (B)"], errors="coerce")
    df["Coût ($) pour 1K token"] = pd.to_numeric(df["Coût ($) pour 1K token"], errors="coerce")

    # Retirer les lignes avec des valeurs manquantes
    df = df.dropna(subset=["Nombre de paramètres (B)", "Coût ($) pour 1K token"])

    # Convertir les paramètres de milliards (B) en unités normales
    df["Nombre de paramètres"] = df["Nombre de paramètres (B)"] * 1e9

    # Ajouter une colonne pour la couleur des points
    df["Type"] = df["Nombre de paramètres"].apply(lambda x: 'SLM' if x <= 7e9 else 'LLM')
    # Régression linéaire
    slope, intercept, r_value, p_value, std_err = linregress(df["Nombre de paramètres"], df["Coût ($) pour 1K token"])
    df["Regression"] = df["Nombre de paramètres"] * slope + intercept

    # Création du nuage de points
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="Nombre de paramètres",
        y="Coût ($) pour 1K token",
        hue="Type",
        palette={'SLM': 'red', 'LLM': 'green'},
        legend='full'
    )

        # Tracer la ligne de régression
    sns.lineplot(
        data=df,
        x="Nombre de paramètres",
        y="Regression",
        color="blue",
        label="Régression linéaire"
    )

    # Ajouter des légendes et un titre
    plt.title("Relation entre le nombre de paramètres et le coût pour 1K tokens", fontsize=14)
    plt.xlabel("Nombre de paramètres", fontsize=12)
    plt.ylabel("Coût ($) pour 1K token", fontsize=12)
    plt.grid(True)

    # Mettre les axes en échelle logarithmique
    plt.xscale('log')
    # plt.yscale('log')

    # Sauvegarder la figure
    plt.savefig("../outputs/cost_vs_params.png")

    # Afficher la figure
    plt.show()

# Exemple d'utilisation
input_csv = "../data/financier.csv"  # Nom du fichier d'entrée
plot_cost_vs_params(input_csv)