import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import numpy as np

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).parent
REPOS_DIR = BASE_DIR / "repositories_analysis"

# Définir vos 4 projets ici avec leurs chemins
PROJECTS = {
    "cinema": {
        "commits": REPOS_DIR / "13_reveler_inegalites_cinema" / "Phase4" / "results" / "commits_detailed.csv",
        "contributors": REPOS_DIR / "13_reveler_inegalites_cinema" / "Phase4" / "results" / "contributors_jobs_inferred.csv"
    },
    "shiftdataportal": {
        "commits": REPOS_DIR / "shiftdataportal" / "Phase4" / "results" / "commits_detailed.csv",
        "contributors": REPOS_DIR / "shiftdataportal" / "Phase4" / "results" / "contributors_jobs_inferred.csv"
    },
    "pollution_eau": {
        "commits": REPOS_DIR / "13_pollution_eau" / "Phase4" / "results" / "commits_detailed.csv",
        "contributors": REPOS_DIR / "13_pollution_eau" / "Phase4" / "results" / "contributors_jobs_inferred.csv"
    },
    "eclaireur_public": {
        "commits": REPOS_DIR / "13_eclaireur_public" / "Phase4" / "results" / "commits_detailed.csv",
        "contributors": REPOS_DIR / "13_eclaireur_public" / "Phase4" / "results" / "contributors_jobs_inferred.csv"
    }
}

# Palette de couleurs pour les métiers
JOB_COLORS = {
    'Data Engineer': '#FF1493',  # Rose flashy
    'Data Scientist': '#00FFFF',  # Cyan flashy
    'Data engineering': '#FF1493',  # Rose flashy (même que Data Engineer)
    'ML Engineer': '#9370DB',  # Violet moyen
    'Backend Engineer': '#4682B4',  # Bleu acier
    'Backend Developer': '#4682B4',  # Bleu acier
    'Backend-oriented': '#5F9EA0',  # Bleu cadet
    'Frontend Engineer': '#90EE90',  # Vert clair
    'Frontend Developer': '#90EE90',  # Vert clair
    'DevOps': '#CD853F',  # Peru
    'Fullstack Engineer': '#D2691E',  # Chocolat
    'Full Stack Developer': '#D2691E',  # Chocolat
    'Data Analyst': '#BA55D3',  # Orchidée moyen
    'Product Manager': '#FF8C00',  # Orange foncé
    'Maintainer': '#8B7355',  # Beige foncé
    'Generalist': '#A0522D',  # Sienna
    'Mixed / support': '#B8860B',  # Goldenrod foncé
    'QA / Test Engineer': '#6B8E23',  # Olive foncé
    'Technical Writer': '#708090',  # Gris ardoise
    'Designer': '#DDA0DD',  # Prune
    'Unknown': '#CCCCCC'  # Gris
}

# Nombre de périodes pour la normalisation (52 = semaines sur 1 an)
TIME_BINS = 52

def normalize_name(name):
    """Normalise les noms pour matcher les contributeurs"""
    if not isinstance(name, str):
        return "unknown"
    return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

def load_project_data(project_name, project_config):
    """Charge les données d'un projet et associe les jobs aux commits"""
    print(f"  Chargement: {project_name}")
    
    # Charger les commits
    commits_df = pd.read_csv(project_config["commits"])
    commits_df['date'] = pd.to_datetime(commits_df['commit_date'], utc=True)
    commits_df['canonical_id'] = commits_df['author_name'].apply(normalize_name)
    
    # Charger les jobs inférés
    jobs_df = pd.read_csv(project_config["contributors"])
    
    # Vérifier si le fichier a la colonne 'job' ou s'il faut l'inférer du 'profile'
    if 'job' not in jobs_df.columns:
        # Pour cinema : utiliser author_email et profile
        if 'author_email' in jobs_df.columns and 'profile' in jobs_df.columns:
            jobs_df['canonical_id'] = jobs_df['author_email'].apply(normalize_name)
            jobs_df['job'] = jobs_df['profile']
            commits_df['canonical_id'] = commits_df['author_email'].apply(normalize_name)
        else:
            print(f"    ✗ Erreur: structure de fichier non reconnue")
            return None
    else:
        jobs_df['canonical_id'] = jobs_df['author_name'].apply(normalize_name)
    
    # Joindre les jobs aux commits
    commits_with_jobs = commits_df.merge(
        jobs_df[['canonical_id', 'job']], 
        on='canonical_id', 
        how='left'
    )
    commits_with_jobs['job'] = commits_with_jobs['job'].fillna('Unknown')
    
    # Informations sur le projet
    min_date = commits_with_jobs['date'].min()
    max_date = commits_with_jobs['date'].max()
    duration_days = (max_date - min_date).days
    
    print(f"    → Durée: {duration_days} jours ({duration_days/365.25:.1f} ans)")
    print(f"    → Commits: {len(commits_with_jobs)}")
    
    return commits_with_jobs

def normalize_timeline(df, bins=52):
    """
    Normalise la timeline d'un projet sur 1 an
    bins: nombre de périodes (52 = semaines)
    """
    df = df.copy()
    min_date = df['date'].min()
    max_date = df['date'].max()
    
    # Calculer la position normalisée (0 à 1)
    df['normalized_time'] = (df['date'] - min_date).dt.total_seconds() / (max_date - min_date).total_seconds()
    
    # Créer des bins pour 1 an
    df['time_bin'] = pd.cut(df['normalized_time'], bins=bins, labels=range(bins))
    df['time_bin'] = df['time_bin'].astype(int)
    
    return df

def aggregate_job_presence(df, bins=52):
    """
    Agrège la présence de chaque métier par période de temps
    Retourne un DataFrame avec les counts par job et par time_bin
    """
    job_counts = df.groupby(['time_bin', 'job']).size().reset_index(name='count')
    
    # Créer un pivot pour avoir jobs en colonnes et time_bins en lignes
    pivot = job_counts.pivot(index='time_bin', columns='job', values='count').fillna(0)
    
    # S'assurer que tous les bins sont présents (0 à bins-1)
    full_index = pd.Index(range(bins), name='time_bin')
    pivot = pivot.reindex(full_index, fill_value=0)
    
    return pivot

def compute_average_evolution(all_projects_data):
    """
    Calcule l'évolution moyenne de chaque métier sur tous les projets
    """
    # Collecter tous les jobs uniques
    all_jobs = set()
    for job_data in all_projects_data.values():
        all_jobs.update(job_data.columns)
    
    print(f"\n  Métiers détectés: {', '.join(sorted(all_jobs))}")
    
    # Créer un DataFrame pour stocker les moyennes
    avg_data = pd.DataFrame(index=range(TIME_BINS))
    
    for job in all_jobs:
        job_series = []
        for project_name, job_data in all_projects_data.items():
            if job in job_data.columns:
                job_series.append(job_data[job])
        
        # Moyenne des projets qui ont ce job
        if job_series:
            avg_data[job] = pd.concat(job_series, axis=1).mean(axis=1)
    
    return avg_data

def plot_average_evolution(avg_data, output_path):
    """
    Crée UN SEUL graphique avec la moyenne de l'évolution de chaque métier
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Trier les métiers par activité totale (pour la légende)
    job_totals = avg_data.sum().sort_values(ascending=False)
    
    # Tracer chaque métier
    for job in job_totals.index:
        ax.plot(
            avg_data.index,
            avg_data[job],
            label=f"{job} ({job_totals[job]:.0f} commits moy.)",
            color=JOB_COLORS.get(job, '#999999'),
            linewidth=2.5,
            alpha=0.85,
            marker='o',
            markersize=3,
            markevery=4  # Marque tous les 4 points pour ne pas surcharger
        )
    
    # Style du graphique
    ax.set_title(
        'Évolution moyenne des métiers sur tous les projets\n(Timeline normalisée sur 1 an)',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel('Temps (normalisé sur 1 an)', fontsize=13)
    ax.set_ylabel('Nombre moyen de commits par période', fontsize=13)
    
    # Légende
    ax.legend(
        loc='upper left',
        fontsize=10,
        framealpha=0.95,
        edgecolor='gray'
    )
    
    # Grille
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Formatter l'axe X (en mois)
    num_labels = 13  # 0 à 12 mois
    tick_positions = np.linspace(0, TIME_BINS-1, num_labels)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f'M{int(i)}' for i in range(num_labels)])
    
    # Améliorer le style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Graphique sauvegardé: {output_path}")
    plt.close()

def generate_summary_statistics(avg_data, all_projects_data, output_path):
    """
    Génère un fichier CSV avec les statistiques détaillées
    """
    stats = []
    
    for job in avg_data.columns:
        # Statistiques moyennes
        avg_total = avg_data[job].sum()
        avg_max = avg_data[job].max()
        avg_mean = avg_data[job].mean()
        
        # Pic d'activité (semaine)
        peak_week = avg_data[job].idxmax()
        peak_value = avg_data[job].max()
        
        # Compter dans combien de projets ce job apparaît
        projects_with_job = sum(1 for data in all_projects_data.values() if job in data.columns)
        
        stats.append({
            'Job': job,
            'Avg Total Commits': round(avg_total, 1),
            'Avg Max (semaine)': round(avg_max, 1),
            'Avg Mean (semaine)': round(avg_mean, 2),
            'Peak Week': f'M{int(peak_week/4.33)}',  # Convertir en mois
            'Peak Value': round(peak_value, 1),
            'Present in N Projects': projects_with_job
        })
    
    stats_df = pd.DataFrame(stats)
    stats_df = stats_df.sort_values('Avg Total Commits', ascending=False)
    stats_df.to_csv(output_path, index=False)
    
    print(f"✅ Statistiques sauvegardées: {output_path}")
    
    return stats_df

def print_summary(stats_df, num_projects):
    """
    Affiche un résumé dans la console
    """
    print("\n" + "="*70)
    print("RÉSUMÉ DE L'ANALYSE")
    print("="*70)
    print(f"Projets analysés: {num_projects}")
    print(f"Métiers détectés: {len(stats_df)}")
    print(f"\nTop 5 des métiers par activité moyenne:")
    print("-"*70)
    
    for idx, row in stats_df.head(5).iterrows():
        print(f"{row['Job']:25} | {row['Avg Total Commits']:6.1f} commits | "
              f"Pic en {row['Peak Week']:4} ({row['Peak Value']:5.1f})")
    
    print("="*70)

def main():
    print("\n" + "="*70)
    print("VISUALISATION DE L'ÉVOLUTION MOYENNE DES MÉTIERS")
    print("="*70 + "\n")
    
    # Configuration matplotlib
    plt.style.use('seaborn-v0_8-whitegrid')
    
    all_projects_data = {}
    
    # Charger et traiter chaque projet
    print("Chargement des données...")
    for project_name, project_config in PROJECTS.items():
        try:
            # Charger les données
            df = load_project_data(project_name, project_config)
            
            # Normaliser la timeline sur 1 an
            df = normalize_timeline(df, bins=TIME_BINS)
            
            # Agréger par métier
            job_evolution = aggregate_job_presence(df, bins=TIME_BINS)
            
            all_projects_data[project_name] = job_evolution
            
        except FileNotFoundError as e:
            print(f"    ✗ Erreur: fichier non trouvé - {e}")
        except Exception as e:
            print(f"    ✗ Erreur: {e}")
    
    if not all_projects_data:
        print("\n❌ Aucune donnée chargée. Vérifiez vos chemins de fichiers.")
        return
    
    print(f"\n✅ {len(all_projects_data)} projets chargés avec succès")
    
    # Calculer les moyennes
    print("\nCalcul des moyennes...")
    avg_data = compute_average_evolution(all_projects_data)
    
    # Créer le dossier de sortie
    output_dir = BASE_DIR / "results"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Générer le graphique
    print("\nGénération du graphique...")
    plot_average_evolution(
        avg_data,
        output_dir / "jobs_evolution_average.png"
    )
    
    # Générer les statistiques
    print("\nGénération des statistiques...")
    stats_df = generate_summary_statistics(
        avg_data,
        all_projects_data,
        output_dir / "jobs_statistics_average.csv"
    )
    
    # Afficher le résumé
    print_summary(stats_df, len(all_projects_data))
    
    print("\n✅ Traitement terminé avec succès!\n")

if __name__ == "__main__":
    main()