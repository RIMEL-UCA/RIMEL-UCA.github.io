import json
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

print("=" * 60)
print("DÉMARRAGE DU SCRIPT DE VISUALISATION")
print("=" * 60)

# Configuration du style
print("\n[1/8] Configuration du style matplotlib...")
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 9
print("✓ Style configuré")

# Création de la figure avec plusieurs subplots
print("\n[2/8] Création de la figure...")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
print("✓ Figure créée (16x10)")

results_dir = Path("results")
print(f"\n[INFO] Répertoire des résultats: {results_dir.absolute()}")
if not results_dir.exists():
    print(f"❌ ERREUR: Le répertoire {results_dir} n'existe pas!")
    exit(1)
print("✓ Répertoire trouvé")

# 1. Distribution des extensions (Top 10) - Barres horizontales
print("\n[3/8] Création du graphique: Distribution des extensions...")
ax1 = fig.add_subplot(gs[0, :2])
file_dist_path = results_dir / "file_distribution.csv"
print(f"   Lecture de: {file_dist_path}")
if not file_dist_path.exists():
    print(f"   ❌ ERREUR: {file_dist_path} n'existe pas!")
else:
    with open(file_dist_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        extensions = [(row["extension"], int(row["count"])) for row in reader]
        extensions.sort(key=lambda x: x[1], reverse=True)
        top_extensions = extensions[:10]
        
        print(f"   → {len(extensions)} extensions trouvées")
        print(f"   → Top 10: {[e[0] for e in top_extensions]}")
        
        exts, counts = zip(*top_extensions)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(exts)))
        ax1.barh(exts, counts, color=colors)
        ax1.set_xlabel("Nombre de fichiers")
        ax1.set_title("Top 10 des extensions de fichiers", fontweight='bold', fontsize=11)
        ax1.invert_yaxis()
        for i, v in enumerate(counts):
            ax1.text(v + max(counts)*0.01, i, str(v), va='center')
    print("   ✓ Graphique créé")

# 2. Répartition globale - Pie chart
print("\n[4/8] Création du graphique: Pie chart...")
ax2 = fig.add_subplot(gs[0, 2])
total_top = sum(counts)
total_all = sum([c for _, c in extensions])
other_count = total_all - total_top

print(f"   → Total fichiers: {total_all}")
print(f"   → Dans top 10: {total_top}")

if other_count > 0:
    pie_labels = list(exts[:5]) + ['Autres']
    pie_values = list(counts[:5]) + [sum(counts[5:]) + other_count]
else:
    pie_labels = exts
    pie_values = counts

colors_pie = plt.cm.Set3(np.linspace(0, 1, len(pie_labels)))
ax2.pie(pie_values, labels=pie_labels, autopct='%1.1f%%', startangle=90, colors=colors_pie)
ax2.set_title("Répartition des types", fontweight='bold', fontsize=11)
print("   ✓ Pie chart créé")

# 3. Distribution des tailles de fichiers de données
print("\n[5/8] Création du graphique: Distribution des tailles...")
ax3 = fig.add_subplot(gs[1, 0])
data_files_path = results_dir / "data_files.csv"
print(f"   Lecture de: {data_files_path}")
if not data_files_path.exists():
    print(f"   ❌ ERREUR: {data_files_path} n'existe pas!")
else:
    with open(data_files_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        sizes = [int(row["size_bytes"]) for row in reader]
        
    print(f"   → {len(sizes)} fichiers de données trouvés")
    if sizes:
        sizes_kb = [s/1024 for s in sizes]
        print(f"   → Taille min: {min(sizes_kb):.2f} KB")
        print(f"   → Taille max: {max(sizes_kb):.2f} KB")
        print(f"   → Taille moyenne: {np.mean(sizes_kb):.2f} KB")
        ax3.hist(sizes_kb, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax3.set_xlabel("Taille (KB)")
        ax3.set_ylabel("Nombre de fichiers")
        ax3.set_title("Distribution des tailles\n(fichiers de données)", fontweight='bold', fontsize=11)
        ax3.set_yscale('log')
        print("   ✓ Histogramme créé")
    else:
        print("   ⚠ Aucun fichier de données trouvé")

# 4. Taille par type de fichier (Box plot)
print("\n[6/8] Création du graphique: Box plot...")
ax4 = fig.add_subplot(gs[1, 1])
with open(data_files_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    data_by_ext = {}
    for row in reader:
        ext = row["extension"] if row["extension"] else "NO_EXT"
        size_kb = int(row["size_bytes"]) / 1024
        if ext not in data_by_ext:
            data_by_ext[ext] = []
        data_by_ext[ext].append(size_kb)

print(f"   → {len(data_by_ext)} types d'extensions avec données")
if data_by_ext:
    box_data = [data_by_ext[ext] for ext in data_by_ext if len(data_by_ext[ext]) > 0]
    box_labels = [ext for ext in data_by_ext if len(data_by_ext[ext]) > 0]
    bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax4.set_ylabel("Taille (KB)")
    ax4.set_title("Tailles par extension\n(fichiers de données)", fontweight='bold', fontsize=11)
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_yscale('log')
    print("   ✓ Box plot créé")

# 5. Profil du repository - Radar chart
print("\n[7/8] Création du graphique: Radar chart...")
ax5 = fig.add_subplot(gs[1, 2], projection='polar')
stats_path = results_dir / "repo_stats_summary.json"
print(f"   Lecture de: {stats_path}")
if not stats_path.exists():
    print(f"   ❌ ERREUR: {stats_path} n'existe pas!")
else:
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    
    print(f"   → Stats: {stats}")
    
    categories = ['Data Files', 'Raw Data', 'Python Code', 'Notebooks', 'Data Formats']
    values = [
        stats.get("data_file_ratio", 0),
        stats.get("data_raw_ratio", 0),
        stats.get("data_code_ratio", 0),
        stats.get("notebooks_ratio", 0),
        stats.get("data_formats_count", 0) / 10  # Normalisé
    ]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    ax5.plot(angles, values, 'o-', linewidth=2, color='darkblue')
    ax5.fill(angles, values, alpha=0.25, color='skyblue')
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(categories, size=8)
    ax5.set_ylim(0, 1)
    ax5.set_title("Profil du Repository", fontweight='bold', fontsize=11, pad=20)
    ax5.grid(True)
    print("   ✓ Radar chart créé")

# 6. Indicateurs clés (KPI Cards)
print("\n[8/8] Création des KPIs...")
ax6 = fig.add_subplot(gs[2, :])
ax6.axis('off')

profile_path = results_dir / "repo_profile.json"
print(f"   Lecture de: {profile_path}")
if not profile_path.exists():
    print(f"   ❌ ERREUR: {profile_path} n'existe pas!")
else:
    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    print(f"   → Profile: {profile}")

    kpis = [
        ("Fichiers totaux", profile.get("total_files", 0), "darkblue"),
        ("Fichiers de données", profile.get("data_files_count", 0), "green"),
        ("Ratio data", f"{profile.get('data_file_ratio', 0):.1%}", "orange"),
        ("Notebooks", stats.get("notebooks_ratio", 0) * profile.get("total_files", 0), "purple"),
        ("Formats data", stats.get("data_formats_count", 0), "red")
    ]

    n_kpis = len(kpis)
    kpi_width = 0.18
    kpi_height = 0.8
    kpi_spacing = 0.02

    for i, (label, value, color) in enumerate(kpis):
        x = i * (kpi_width + kpi_spacing) + 0.05
        
        # Rectangle de fond
        rect = plt.Rectangle((x, 0.1), kpi_width, kpi_height, 
                             facecolor=color, alpha=0.2, 
                             edgecolor=color, linewidth=2,
                             transform=ax6.transAxes)
        ax6.add_patch(rect)
        
        # Valeur
        ax6.text(x + kpi_width/2, 0.55, str(value), 
                ha='center', va='center', fontsize=16, 
                fontweight='bold', color=color,
                transform=ax6.transAxes)
        
        # Label
        ax6.text(x + kpi_width/2, 0.25, label, 
                ha='center', va='center', fontsize=9,
                color='black', transform=ax6.transAxes)
    print("   ✓ KPIs créés")

# Titre général
fig.suptitle('Dashboard d\'Analyse du Repository', 
            fontsize=16, fontweight='bold', y=0.98)

# Sauvegarde
print("\n" + "=" * 60)
print("SAUVEGARDE DU DASHBOARD")
print("=" * 60)
output_file = 'repo_dashboard.png'
print(f"Fichier de sortie: {output_file}")
print("Sauvegarde en cours...")
plt.savefig(output_file, dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print(f"✓ Dashboard sauvegardé avec succès dans '{output_file}'")
print(f"  → Résolution: 300 DPI")
print(f"  → Taille: ~{Path(output_file).stat().st_size / 1024 / 1024:.2f} MB")
plt.close()
print("\n" + "=" * 60)
print("TERMINÉ AVEC SUCCÈS !")
print("=" * 60)