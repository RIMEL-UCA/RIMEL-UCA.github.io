import pandas as pd
import warnings
from plotnine.exceptions import PlotnineWarning
from plotnine import (
    ggplot, aes, geom_point, geom_smooth,
    labs, theme_minimal
)
from pathlib import Path

CONTRIBUTORS_CSV = "2-nombre-contributeurs/data/contributors.csv"
COMMITS_TYPES_CSV = "3-activite-contributeurs/data/commits_types.csv"
OUTPUT_DIR = Path("3-activite-contributeurs/outputs/graphs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Charger les données
df_contributors = pd.read_csv(CONTRIBUTORS_CSV)
df_commits = pd.read_csv(COMMITS_TYPES_CSV)

# Fusionner les deux datasets sur le nom du repo
warnings.filterwarnings("ignore", category=PlotnineWarning)
 
# Fusionner les deux datasets sur le nom du repo
df = pd.merge(df_commits, df_contributors, on="repo", how="inner")

print(f"Analyse de {len(df)} repos")

# Calculer les ratios
df["ratio_fix"] = df["fix"] / df["total_commits"]

df_refactor = df[df["feat"] > 0].copy()
df_refactor["ratio_refactor_feat"] = (
    df_refactor["refactor"] / df_refactor["feat"]
)

# Graphique 1: Ratio fix vs contributeurs
plot_fix = (
    ggplot(df, aes(x="contributors", y="ratio_fix"))
    + geom_point(alpha=0.7)
    + geom_smooth(method="lm", se=True)
    + theme_minimal()
    + labs(
        title="Ratio fix / total commits selon le nombre de contributeurs",
        x="Nombre de contributeurs",
        y="Ratio fix / total commits"
    )
)

plot_fix.save(
    OUTPUT_DIR / "ratio_fix_vs_contributeurs.png",
    width=8,
    height=5,
    dpi=300
)

# Graphique 2: Ratio refactor/feat vs contributeurs
plot_refactor = (
    ggplot(df_refactor, aes(
        x="contributors",
        y="ratio_refactor_feat"
    ))
    + geom_point(alpha=0.7)
    + geom_smooth(method="lm", se=True)
    + theme_minimal()
    + labs(
        title="Ratio refactor / feat selon le nombre de contributeurs",
        x="Nombre de contributeurs",
        y="Ratio refactor / feat"
    )
)

plot_refactor.save(
    OUTPUT_DIR / "ratio_refactor_feat_vs_contributeurs.png",
    width=8,
    height=5,
    dpi=300
)

print("Graphes generes avec succes.")

