import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Chemin vers le fichier CSV généré
csv_path = Path("results/access_churn_shiftdataportal.csv")
df = pd.read_csv(csv_path)

# Conversion et tri par date
df["month"] = pd.to_datetime(df["month"])
df = df.sort_values("month")

# Labels adaptés pour le Shift Data Portal
labels = {
    "raw_data_files": "Données brutes",
    "data_preparation_scripts": "Scripts de préparation",
    "backend_api": "API Backend (GraphQL)",
    "frontend_client": "Frontend (React)",
    "infrastructure_deployment": "Infra / CI-CD",
    "documentation": "Documentation",
}

series = list(labels.keys())

# Couleurs cohérentes pour les deux graphiques
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#34495e']

# Création de deux sous-graphiques
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# === Graphique 1 : Lignes multiples ===
for idx, col in enumerate(series):
    axes[0].plot(
        df["month"],
        df[col],
        marker="o",
        label=labels[col],
        color=colors[idx],
        linewidth=2,
        markersize=4
    )

axes[0].set_title(
    "Évolution mensuelle de l'activité de développement - The Shift Data Portal",
    fontsize=13,
    fontweight='bold',
    pad=15
)
axes[0].set_ylabel("Nombre de commits", fontsize=11)
axes[0].legend(loc="upper left", fontsize=9, framealpha=0.9)
axes[0].grid(True, linestyle="--", alpha=0.3)

# === Graphique 2 : Aires empilées ===
axes[1].stackplot(
    df["month"],
    [df[col] for col in series],
    labels=[labels[col] for col in series],
    colors=colors,
    alpha=0.85
)

axes[1].set_title(
    "Répartition de l'effort de développement par composant",
    fontsize=13,
    fontweight='bold',
    pad=15
)
axes[1].set_ylabel("Nombre de commits", fontsize=11)
axes[1].set_xlabel("Mois", fontsize=11)
axes[1].legend(loc="upper left", fontsize=9, framealpha=0.9)
axes[1].grid(True, linestyle="--", alpha=0.3)

# Sauvegarde
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)
plt.tight_layout()
plt.savefig(output_dir / "shiftdataportal_churn_analysis.png", dpi=200)
print(f"[✓] Graphique double sauvegardé dans {output_dir / 'shiftdataportal_churn_analysis.png'}")

plt.show()