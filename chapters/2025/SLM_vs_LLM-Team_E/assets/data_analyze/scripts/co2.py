import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

def filter_co2_by_params(input_csv, output_csv):
    # Lire le fichier CSV d'entrée
    df = pd.read_csv(input_csv)

    # Filtrer les lignes avec Type=💬 chat models (RLHF, DPO, IFT, ...)
    # df = df[df["Type"] == "💬 chat models (RLHF, DPO, IFT, ...)"]
    # df=df[df["Type"].str.contains("🟢 pretrained")]

    # S'assurer que les colonnes nécessaires existent
    required_columns = ["#Params (B)", "CO₂ cost (kg)"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"La colonne '{col}' est manquante dans le fichier CSV d'entrée.")

    # Convertir les paramètres de milliards (B) en unités normales
    df["#Params"] = df["#Params (B)"] * 1e9

    # Filtrer uniquement les colonnes pertinentes
    co2_by_params_df = df[["#Params", "CO₂ cost (kg)"]]

    # Sauvegarder le nouveau CSV
    co2_by_params_df.to_csv(output_csv, index=False)

    print(f"Le fichier filtré a été sauvegardé sous le nom '{output_csv}'.")

    # Régression linéaire
    slope, intercept, r_value, p_value, std_err = linregress(co2_by_params_df["#Params"], co2_by_params_df["CO₂ cost (kg)"])
    co2_by_params_df["Regression"] = co2_by_params_df["#Params"] * slope + intercept

    # Création de la courbe pour 'CO₂ cost (kg)' et la ligne de régression
    plt.figure(figsize=(10, 6))

    # Colorier les points en fonction du nombre de paramètres
    co2_by_params_df["Type"] = co2_by_params_df["#Params"].apply(lambda x: 'SLM' if x <= 7e9 else 'LLM')
    sns.scatterplot(
        data=co2_by_params_df,
        x="#Params",
        y="CO₂ cost (kg)",
        hue="Type",
        palette={'SLM': 'red', 'LLM': 'green'},
        legend=True
    )

    sns.lineplot(
        data=co2_by_params_df,
        x="#Params",
        y="Regression",
        color="blue",
        label="Régression linéaire"
    )

    # Ajouter des légendes et un titre
    plt.title("Relation entre le nombre de paramètres et le coût en CO₂", fontsize=14)
    plt.xlabel("Nombre de paramètres", fontsize=12)
    plt.ylabel("Coût en CO₂ (kg)", fontsize=12)
    plt.legend(title="Type")
    plt.grid(True)

    # Mettre les axes en échelle logarithmique
    plt.xscale('log')
    # plt.yscale('log')

    # Sauvegarder la figure
    plt.savefig("../outputs/o2_by_params.png")

    # Afficher la figure
    plt.show()

# Exemple d'utilisation
input_csv = "../data/leaderboard.csv"  # Nom du fichier d'entrée
output_csv = "../outputs/co2_by_params.csv"  # Nom du fichier de sortie
filter_co2_by_params(input_csv, output_csv)