import pandas as pd
import warnings
from plotnine.exceptions import PlotnineWarning
from plotnine import (
    ggplot, aes, geom_bar, geom_hline, geom_boxplot,
    labs, theme_minimal, theme, element_text, scale_x_discrete
)
from pathlib import Path

SUMMARY_CSV = "1-qualite/outputs/summary.csv"
REPOS_GROUPS_CSV = "2-nombre-contributeurs/repos_groups.csv"
CONTRIBUTORS_CSV = "2-nombre-contributeurs/data/contributors.csv"
OUTPUT_DIR = Path("2-nombre-contributeurs/graphs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df_summary = pd.read_csv(SUMMARY_CSV)
df_groups = pd.read_csv(REPOS_GROUPS_CSV)
df_contributors = pd.read_csv(CONTRIBUTORS_CSV)

warnings.filterwarnings("ignore", category=PlotnineWarning)

df_summary["repo_name"] = df_summary["repo_url"].str.extract(
    r"dataforgoodfr/(.+)$"
)[0]

# Merge: keep only repos that have a group and a contributor count
df = pd.merge(df_summary, df_groups, on="repo_name", how="inner")
df = pd.merge(df, df_contributors, left_on="repo_name", right_on="repo", how="inner")

# Compute dynamic labels for each repo_group based on contributor ranges
group_labels = {}
groups = []
for g in sorted(df["repo_group"].unique()):
    df_g = df[df["repo_group"] == g]
    if df_g.empty:
        continue
    mn = int(df_g["contributors"].min())
    mx = int(df_g["contributors"].max())
    if mn == mx:
        label = f"Groupe {g} ({mn} contributeurs)"
    else:
        label = f"Groupe {g} ({mn}-{mx} contributeurs)"
    group_labels[g] = label
    groups.append((g, label))

# Map repo_group -> label on dataframe
df["group_label"] = df["repo_group"].map(group_labels)

global_median = df["score"].median()

for group_num, group_label in groups:
    df_group = df[df["repo_group"] == group_num].copy()
    
    if len(df_group) == 0:
        print(f"Aucune donnée pour le {group_label}")
        continue
    
    median_score = df_group["score"].median()
    print(f"{group_label}: {len(df_group)} repos, médiane = {median_score:.2f}")
    df_group = df_group.sort_values("score")
    
    plot = (
        ggplot(df_group, aes(x="repo_name", y="score"))
        + geom_bar(stat="identity", fill="steelblue", alpha=0.7)
        + geom_hline(yintercept=median_score, color="red", linetype="dashed", size=1)
        + geom_hline(yintercept=global_median, color="orange", linetype="dashed", size=1)
        + theme_minimal()
        + theme(
            axis_text_x=element_text(rotation=45, hjust=1, size=8),
            figure_size=(12, 6)
        )
        + labs(
            title=f"Score de qualité par rapport à la médiane pour les dépôts de {group_label.split('(')[1].split(')')[0]}",
            x="Nom du dépôt",
            y="Score de qualité",
            caption=f"Ligne rouge: médiane du groupe = {median_score:.2f} | Ligne orange: médiane globale = {global_median:.2f}"
        )
    )
    
    filename = f"qualite_groupe_{group_num}.png"
    plot.save(
        OUTPUT_DIR / filename,
        width=12,
        height=6,
        dpi=300
    )
    print(f"Graphique: {filename}")

print("Génération du diagramme en boîtes")
df_boxplot = df.copy()
df_boxplot["group_label"] = pd.Categorical(
    df_boxplot["group_label"],
    categories=[label for _, label in groups],
    ordered=True
)

for group_num, group_label in groups:
    df_group_stats = df_boxplot[df_boxplot["repo_group"] == group_num]
    if len(df_group_stats) > 0:
        print(f"{group_label}: min={df_group_stats['score'].min():.2f}, med={df_group_stats['score'].median():.2f}, max={df_group_stats['score'].max():.2f}")

boxplot = (
    ggplot(df_boxplot, aes(x="group_label", y="score", fill="group_label"))
    + geom_boxplot(alpha=0.7, show_legend=False)
    + theme_minimal()
    + theme(
        axis_text_x=element_text(rotation=15, hjust=1, size=10),
        figure_size=(10, 6)
    )
    + labs(
        title="Distribution des scores de qualité par groupe de contributeurs",
        x="Groupe de dépôts",
        y="Score de qualité",
    )
)

boxplot_filename = "qualite_boxplot_groupes.png"
boxplot.save(
    OUTPUT_DIR / boxplot_filename,
    width=10,
    height=6,
    dpi=300
)
print(f"Diagramme en boîtes: {boxplot_filename}")
print("Graphiques écrits dans", OUTPUT_DIR)
