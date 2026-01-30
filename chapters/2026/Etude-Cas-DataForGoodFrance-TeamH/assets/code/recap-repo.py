import json
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys

print("=" * 70)
print("DASHBOARD COMPARATIF MULTI-REPOSITORIES")
print("=" * 70)

# Configuration
REPO_PATHS = [
    "repositories_analysis/shiftdataportal/Phase1/results",
    "repositories_analysis/13_pollution_eau/Phase1/results",
    "repositories_analysis/13_reveler_inegalites_cinema/Phase1/results",
    "repositories_analysis/13_eclaireur_public/Phase1/results"
]

# Ou passez les chemins en arguments
if len(sys.argv) > 1:
    REPO_PATHS = sys.argv[1:]
    print(f"\n[INFO] {len(REPO_PATHS)} chemins fournis en arguments")
else:
    print("\n[INFO] Utilisation des chemins définis dans le script")

if not REPO_PATHS:
    print("\n❌ ERREUR: Aucun chemin de repository fourni!")
    print("\nUsage:")
    print("  python multi_repo_dashboard.py /path/to/repo1/results /path/to/repo2/results ...")
    print("\nOu modifiez la variable REPO_PATHS dans le script")
    sys.exit(1)

print(f"\n[INFO] Analyse de {len(REPO_PATHS)} repositories")

# Configuration du style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 9

# Création de la figure
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

# Structure pour stocker les données de tous les repos
repos_data = []

print("\n" + "=" * 70)
print("CHARGEMENT DES DONNÉES")
print("=" * 70)

for idx, repo_path in enumerate(REPO_PATHS):
    print(f"\n[{idx+1}/{len(REPO_PATHS)}] Repository: {repo_path}")
    
    repo_path = Path(repo_path)
    # Extraire le premier élément du chemin (nom du repo)
    # Ex: "SHIFTPORTAL/1/results" -> "SHIFTPORTAL"
    path_parts = repo_path.parts
    repo_name = path_parts[0] if len(path_parts) > 0 else repo_path.name
    
    print(f"   → Nom du repo: {repo_name}")
    
    if not repo_path.exists():
        print(f"   ⚠ ATTENTION: {repo_path} n'existe pas, ignoré")
        continue
    
    repo_info = {
        "name": repo_name, 
        "path": repo_path, 
        "num": idx + 1,  # Numéro du repo (1, 2, 3, 4)
        "label": f"[{idx + 1}]"  # Label court pour les graphiques
    }
    
    # Charger repo_profile.json
    profile_path = repo_path / "repo_profile.json"
    if profile_path.exists():
        with open(profile_path, "r", encoding="utf-8") as f:
            repo_info["profile"] = json.load(f)
        print(f"   ✓ Profile chargé: {repo_info['profile'].get('total_files', 0)} fichiers")
    else:
        print(f"   ⚠ repo_profile.json non trouvé")
        repo_info["profile"] = {}
    
    # Charger repo_stats_summary.json
    stats_path = repo_path / "repo_stats_summary.json"
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            repo_info["stats"] = json.load(f)
        print(f"   ✓ Stats chargées")
    else:
        print(f"   ⚠ repo_stats_summary.json non trouvé")
        repo_info["stats"] = {}
    
    # Charger file_distribution.csv
    dist_path = repo_path / "file_distribution.csv"
    if dist_path.exists():
        with open(dist_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            repo_info["extensions"] = [(row["extension"], int(row["count"])) for row in reader]
        print(f"   ✓ Distribution chargée: {len(repo_info['extensions'])} extensions")
    else:
        print(f"   ⚠ file_distribution.csv non trouvé")
        repo_info["extensions"] = []
    
    # Charger data_files.csv
    data_path = repo_path / "data_files.csv"
    if data_path.exists():
        with open(data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            repo_info["data_files"] = list(reader)
        print(f"   ✓ Fichiers de données chargés: {len(repo_info['data_files'])} fichiers")
    else:
        print(f"   ⚠ data_files.csv non trouvé")
        repo_info["data_files"] = []
    
    repos_data.append(repo_info)

if not repos_data:
    print("\n❌ ERREUR: Aucune donnée chargée!")
    sys.exit(1)

print(f"\n✓ {len(repos_data)} repositories chargés avec succès")

# ============================================================================
# GRAPHIQUES
# ============================================================================

print("\n" + "=" * 70)
print("CRÉATION DES GRAPHIQUES")
print("=" * 70)

# 1. Comparaison du nombre total de fichiers
print("\n[1/8] Graphique: Comparaison nombre de fichiers...")
ax1 = fig.add_subplot(gs[0, 0])
repo_labels = [r["label"] for r in repos_data]  # [1], [2], [3], [4]
total_files = [r["profile"].get("total_files", 0) for r in repos_data]
colors1 = plt.cm.viridis(np.linspace(0.2, 0.8, len(repo_labels)))
bars1 = ax1.bar(range(len(repo_labels)), total_files, color=colors1)
ax1.set_xticks(range(len(repo_labels)))
ax1.set_xticklabels(repo_labels, rotation=0)
ax1.set_ylabel("Nombre de fichiers")
ax1.set_title("Nombre total de fichiers par repo", fontweight='bold', fontsize=11)
for i, v in enumerate(total_files):
    ax1.text(i, v + max(total_files)*0.02, str(v), ha='center', va='bottom', fontweight='bold')
print("   ✓ Créé")

# 2. Comparaison des ratios (data, code, notebooks)
print("\n[2/8] Graphique: Comparaison des ratios...")
ax2 = fig.add_subplot(gs[0, 1])
x_pos = np.arange(len(repo_labels))
width = 0.25
data_ratios = [r["stats"].get("data_file_ratio", 0) for r in repos_data]
code_ratios = [r["stats"].get("data_code_ratio", 0) for r in repos_data]
notebook_ratios = [r["stats"].get("notebooks_ratio", 0) for r in repos_data]

ax2.bar(x_pos - width, data_ratios, width, label='Data Files', color='steelblue')
ax2.bar(x_pos, code_ratios, width, label='Python Code', color='orange')
ax2.bar(x_pos + width, notebook_ratios, width, label='Notebooks', color='green')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(repo_labels, rotation=0)
ax2.set_ylabel("Ratio")
ax2.set_title("Ratios de fichiers par type", fontweight='bold', fontsize=11)
ax2.legend(loc='upper right', fontsize=8)
print("   ✓ Créé")

# 3. Répartition des extensions (stacked bar)
print("\n[3/8] Graphique: Répartition des extensions...")
ax3 = fig.add_subplot(gs[0, 2])
# Récupérer toutes les extensions uniques
all_extensions = set()
for repo in repos_data:
    for ext, _ in repo["extensions"]:
        all_extensions.add(ext)
all_extensions = sorted(list(all_extensions))[:10]  # Top 10

ext_data = {ext: [] for ext in all_extensions}
for repo in repos_data:
    ext_dict = {e: c for e, c in repo["extensions"]}
    for ext in all_extensions:
        ext_data[ext].append(ext_dict.get(ext, 0))

bottom = np.zeros(len(repo_labels))
colors3 = plt.cm.tab20(np.linspace(0, 1, len(all_extensions)))
for i, ext in enumerate(all_extensions):
    ax3.bar(repo_labels, ext_data[ext], bottom=bottom, label=ext[:10], color=colors3[i])
    bottom += ext_data[ext]

ax3.set_ylabel("Nombre de fichiers")
ax3.set_title("Distribution des extensions", fontweight='bold', fontsize=11)
ax3.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=7)
ax3.tick_params(axis='x', rotation=0)
print("   ✓ Créé")

# 4. Comparaison du nombre de fichiers de données
print("\n[4/8] Graphique: Fichiers de données...")
ax4 = fig.add_subplot(gs[1, 0])
data_files_count = [r["profile"].get("data_files_count", 0) for r in repos_data]
colors4 = plt.cm.plasma(np.linspace(0.2, 0.8, len(repo_labels)))
bars4 = ax4.bar(range(len(repo_labels)), data_files_count, color=colors4)
ax4.set_xticks(range(len(repo_labels)))
ax4.set_xticklabels(repo_labels, rotation=0)
ax4.set_ylabel("Nombre de fichiers de données")
ax4.set_title("Fichiers de données par repo", fontweight='bold', fontsize=11)
for i, v in enumerate(data_files_count):
    ax4.text(i, v + max(data_files_count + [1])*0.02, str(v), ha='center', va='bottom', fontweight='bold')
print("   ✓ Créé")

# 5. Distribution des tailles moyennes
print("\n[5/8] Graphique: Tailles moyennes des fichiers...")
ax5 = fig.add_subplot(gs[1, 1])
avg_sizes = []
for repo in repos_data:
    if repo["data_files"]:
        sizes = [int(f["size_bytes"]) / 1024 for f in repo["data_files"]]
        avg_sizes.append(np.mean(sizes))
    else:
        avg_sizes.append(0)

bars5 = ax5.bar(range(len(repo_labels)), avg_sizes, color='coral')
ax5.set_xticks(range(len(repo_labels)))
ax5.set_xticklabels(repo_labels, rotation=0)
ax5.set_ylabel("Taille moyenne (KB)")
ax5.set_title("Taille moyenne des fichiers de données", fontweight='bold', fontsize=11)
for i, v in enumerate(avg_sizes):
    ax5.text(i, v + max(avg_sizes + [1])*0.02, f"{v:.1f}", ha='center', va='bottom', fontweight='bold')
print("   ✓ Créé")

# 6. Radar chart comparatif
print("\n[6/8] Graphique: Radar comparatif...")
ax6 = fig.add_subplot(gs[1, 2], projection='polar')
categories = ['Data\nFiles', 'Raw\nData', 'Python\nCode', 'Notebooks']
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

colors6 = plt.cm.Set2(np.linspace(0, 1, len(repos_data)))
for idx, repo in enumerate(repos_data):
    values = [
        repo["stats"].get("data_file_ratio", 0),
        repo["stats"].get("data_raw_ratio", 0),
        repo["stats"].get("data_code_ratio", 0),
        repo["stats"].get("notebooks_ratio", 0),
    ]
    values += values[:1]
    ax6.plot(angles, values, 'o-', linewidth=2, label=repo["label"], color=colors6[idx])
    ax6.fill(angles, values, alpha=0.15, color=colors6[idx])

ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(categories, size=7)
ax6.set_ylim(0, max([max([r["stats"].get("data_file_ratio", 0), 
                           r["stats"].get("data_raw_ratio", 0),
                           r["stats"].get("data_code_ratio", 0),
                           r["stats"].get("notebooks_ratio", 0)]) for r in repos_data]) * 1.1)
ax6.set_title("Profils comparés", fontweight='bold', fontsize=11, pad=10, y=1.0)  # Descendre le titre
ax6.legend(loc='upper right', bbox_to_anchor=(1.25, 0.95), fontsize=8)  # Descendre la légende
ax6.grid(True)
print("   ✓ Créé")

# 7. Heatmap des formats de données
print("\n[7/8] Graphique: Heatmap des formats...")
ax7 = fig.add_subplot(gs[2, :2])
data_formats = ['.csv', '.json', '.xlsx', '.parquet', '.sql', '.ipynb']
heatmap_data = []
for repo in repos_data:
    ext_dict = {e: c for e, c in repo["extensions"]}
    row = [ext_dict.get(fmt, 0) for fmt in data_formats]
    heatmap_data.append(row)

heatmap_data = np.array(heatmap_data)
if heatmap_data.max() > 0:
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', 
                xticklabels=data_formats, yticklabels=repo_labels,
                cbar_kws={'label': 'Nombre de fichiers'}, ax=ax7)
    ax7.set_title("Heatmap des formats de données", fontweight='bold', fontsize=11)
    ax7.set_ylabel("")  # Supprimer le label de l'axe Y
    ax7.tick_params(axis='y', rotation=0)  # Assurer que les labels sont horizontaux
print("   ✓ Créé")

# 8. Tableau récapitulatif (KPIs) + LÉGENDE
print("\n[8/8] Création du tableau récapitulatif...")
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('off')

# Créer un tableau avec les métriques clés
table_data = []
table_data.append(['Repo', 'Fichiers', 'Data', 'Code', 'Notebooks'])
for repo in repos_data:
    table_data.append([
        repo["label"],
        str(repo["profile"].get("total_files", 0)),
        str(repo["profile"].get("data_files_count", 0)),
        str(int(repo["stats"].get("data_code_ratio", 0) * repo["profile"].get("total_files", 0))),
        str(int(repo["stats"].get("notebooks_ratio", 0) * repo["profile"].get("total_files", 0)))
    ])

table = ax8.table(cellText=table_data, cellLoc='center', loc='upper center',
                  colWidths=[0.15, 0.2, 0.15, 0.15, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.5)

# Styliser l'en-tête
for i in range(5):
    cell = table[(0, i)]
    cell.set_facecolor('#4CAF50')
    cell.set_text_props(weight='bold', color='white')

# Ajouter la légende en dessous du tableau
legend_y = 0.15  # Position plus basse
ax8.text(0.5, legend_y, "Légende des repositories:", 
         ha='center', va='top', fontweight='bold', fontsize=9,
         transform=ax8.transAxes)

legend_text = "\n".join([f"{repo['label']} = {repo['name']}" for repo in repos_data])
ax8.text(0.5, legend_y - 0.08, legend_text, 
         ha='center', va='top', fontsize=8,
         transform=ax8.transAxes,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, pad=0.5))

ax8.set_title("Résumé des métriques", fontweight='bold', fontsize=11, pad=10)
print("   ✓ Créé")

# 9. Graphique bonus: Top extensions globales
print("\n[BONUS] Graphique: Top extensions globales...")
ax9 = fig.add_subplot(gs[3, :])
all_ext_count = {}
for repo in repos_data:
    for ext, count in repo["extensions"]:
        all_ext_count[ext] = all_ext_count.get(ext, 0) + count

top_global_ext = sorted(all_ext_count.items(), key=lambda x: x[1], reverse=True)[:15]
exts_global, counts_global = zip(*top_global_ext)
colors9 = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(exts_global)))
ax9.barh(exts_global, counts_global, color=colors9)
ax9.set_xlabel("Nombre total de fichiers")
ax9.set_title("Top 15 extensions globales (tous repos confondus)", fontweight='bold', fontsize=11)
ax9.invert_yaxis()
for i, v in enumerate(counts_global):
    ax9.text(v + max(counts_global)*0.01, i, str(v), va='center', fontweight='bold')
print("   ✓ Créé")

# Titre général
fig.suptitle(f'Dashboard Comparatif - {len(repos_data)} Repositories', 
            fontsize=18, fontweight='bold', y=0.995)

# Sauvegarde
print("\n" + "=" * 70)
print("SAUVEGARDE")
print("=" * 70)
output_file = 'multi_repo_dashboard.png'
print(f"Fichier de sortie: {output_file}")
plt.savefig(output_file, dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
file_size = Path(output_file).stat().st_size / 1024 / 1024
print(f"✓ Dashboard sauvegardé avec succès!")
print(f"  → Résolution: 300 DPI")
print(f"  → Taille: {file_size:.2f} MB")
plt.close()

print("\n" + "=" * 70)
print("TERMINÉ AVEC SUCCÈS !")
print("=" * 70)
print(f"\nRésumé:")
for repo in repos_data:
    print(f"  • {repo['name']}: {repo['profile'].get('total_files', 0)} fichiers, "
          f"{repo['profile'].get('data_files_count', 0)} data files")
print(f"\nDashboard: {output_file}")