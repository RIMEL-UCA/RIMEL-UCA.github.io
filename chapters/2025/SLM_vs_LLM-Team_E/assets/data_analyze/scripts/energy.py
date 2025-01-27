import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Fonction pour calculer l'énergie par million de tokens et retourner les moyennes
def calculate_energy_per_million_tokens(input_csv):
    # Lire le fichier CSV d'entrée
    df = pd.read_csv(input_csv)

    # S'assurer que les colonnes nécessaires existent
    required_columns = [
        "Energy (tokens/kWh)",
        "Params (B)"
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"La colonne '{col}' est manquante dans le fichier CSV d'entrée.")

    # Filtrer les valeurs dont le nombre de paramètres est supérieur à 40
    # df = df[df["Params (B)"] < 40]

        # Convertir les paramètres de milliards (B) en unités normales
    df["Params"] = df["Params (B)"] * 1e9

    # Calculer l'énergie consommée pour un million de tokens
    df["Energy per million tokens (kWh)"] = 1e6 / df["Energy (tokens/kWh)"]

    # Calculer la moyenne de 'Energy per million tokens (kWh)' pour chaque modèle
    averages = (
        df.groupby(["Params"])[["Energy per million tokens (kWh)"]]
        .mean()
        .reset_index()
    )

    return averages

# Lire les fichiers CSV et calculer les moyennes
averages_A10 = calculate_energy_per_million_tokens('../data/input-A10.csv')
averages_A100 = calculate_energy_per_million_tokens('../data/input-A100.csv')
averages_T4 = calculate_energy_per_million_tokens('../data/input-T4.csv')
averages_32vCPU = calculate_energy_per_million_tokens('../data/input-32vCPU.csv')

# Création de la courbe pour 'Energy per million tokens (kWh)'
plt.figure(figsize=(12, 8))

sns.lineplot(
    data=averages_A10,
    x="Params",
    y="Energy per million tokens (kWh)",
    marker='o',
    label="A10"
)

sns.lineplot(
    data=averages_A100,
    x="Params",
    y="Energy per million tokens (kWh)",
    marker='o',
    label="A100"
)

sns.lineplot(
    data=averages_T4,
    x="Params",
    y="Energy per million tokens (kWh)",
    marker='o',
    label="T4"
)

sns.lineplot(
    data=averages_32vCPU,
    x="Params",
    y="Energy per million tokens (kWh)",
    marker='o',
    label="32vCPU"
)

# Dessiner une droite verticale à x=7
plt.axvline(x=7*1e9, color='r', linestyle='--', label='x = 7')

# Ajouter des légendes et un titre
plt.title("Energie consommée en fonction du nombre de parametres du model (pour un million de tokens)", fontsize=14)
plt.xlabel("Params", fontsize=12)
plt.ylabel("Energy (kWh)", fontsize=12)
plt.legend()
plt.grid(True)

# Mettre les axes en échelle logarithmique
plt.xscale('log')
plt.yscale('log')

# Sauvegarder la figure
plt.savefig("../outputs/energy_per_million_tokens_vs_params.png")
