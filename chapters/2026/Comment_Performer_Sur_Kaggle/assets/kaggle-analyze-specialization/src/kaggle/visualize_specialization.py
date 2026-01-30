#!/usr/bin/env python3
"""
Visualisation de la spécialisation des top compétiteurs Kaggle
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import numpy as np

# Configuration de style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

TAGS_OF_INTEREST = ["NLP", "Time Series Analysis", "Tabular"]


def load_data():
    """Charge les données de spécialisation."""
    logger.info("Chargement des données...")
    df = pd.read_csv('../../data/specialization_matrix.csv')
    logger.success(f"Données chargées : {len(df)} compétiteurs")
    return df


def plot_top_competitors_heatmap(df, top_n=50):
    """Crée une heatmap des top N compétiteurs."""
    logger.info(f"Création de la heatmap pour le top {top_n}")

    # Sélectionner le top N
    top_competitors = df.head(top_n)

    # Créer une matrice avec les pourcentages uniquement
    percentage_cols = [f'{tag}_percentage' for tag in TAGS_OF_INTEREST]
    heatmap_data = top_competitors[percentage_cols].values

    # Créer la heatmap
    fig, ax = plt.subplots(figsize=(10, max(12, top_n * 0.3)))

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        xticklabels=TAGS_OF_INTEREST,
        yticklabels=top_competitors['username'].values,
        cbar_kws={'label': 'Pourcentage (%)'},
        ax=ax
    )

    plt.title(f'Matrice de Spécialisation - Top {top_n} Compétiteurs', fontsize=16, fontweight='bold')
    plt.xlabel('Domaine', fontsize=12)
    plt.ylabel('Compétiteur', fontsize=12)
    plt.tight_layout()
    plt.savefig('../../data/heatmap_specialization.png', dpi=300, bbox_inches='tight')
    logger.success("Heatmap sauvegardée : ../../data/heatmap_specialization.png")
    plt.close()


def plot_specialization_distribution(df):
    """Distribution de la spécialisation par domaine."""
    logger.info("Création des distributions de spécialisation")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, tag in enumerate(TAGS_OF_INTEREST):
        percentage_col = f'{tag}_percentage'

        axes[i].hist(df[percentage_col], bins=20, edgecolor='black', alpha=0.7,
                     color=['#FF6B6B', '#4ECDC4', '#45B7D1'][i])
        axes[i].axvline(df[percentage_col].mean(), color='red', linestyle='--', linewidth=2,
                        label=f'Moyenne: {df[percentage_col].mean():.1f}%')
        axes[i].set_xlabel('Pourcentage de spécialisation (%)', fontsize=11)
        axes[i].set_ylabel('Nombre de compétiteurs', fontsize=11)
        axes[i].set_title(f'Distribution - {tag}', fontsize=12, fontweight='bold')
        axes[i].legend()
        axes[i].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('../../data/distribution_specialization.png', dpi=300, bbox_inches='tight')
    logger.success("Distributions sauvegardées : ../../data/distribution_specialization.png")
    plt.close()


def plot_average_specialization_bars(df):
    """Barres des moyennes de spécialisation."""
    logger.info("Création du graphique en barres des moyennes")

    averages = []
    for tag in TAGS_OF_INTEREST:
        avg = df[f'{tag}_percentage'].mean()
        averages.append(avg)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax.bar(TAGS_OF_INTEREST, averages, color=colors, edgecolor='black', linewidth=1.5)

    # Ajouter les valeurs sur les barres
    for bar, avg in zip(bars, averages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{avg:.1f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Pourcentage moyen de spécialisation (%)', fontsize=12)
    ax.set_title('Spécialisation Moyenne par Domaine', fontsize=16, fontweight='bold')
    ax.set_ylim(0, max(averages) * 1.2)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('../../data/average_specialization.png', dpi=300, bbox_inches='tight')
    logger.success("Graphique en barres sauvegardé : ../../data/average_specialization.png")
    plt.close()


def plot_top_vs_rest(df, threshold=10):
    """Compare les tops (>= threshold participations) vs le reste."""
    logger.info(f"Comparaison top (>={threshold} participations) vs reste")

    top_df = df[df['total_competitions'] >= threshold]
    rest_df = df[df['total_competitions'] < threshold]

    logger.info(f"Top compétiteurs : {len(top_df)}")
    logger.info(f"Reste : {len(rest_df)}")

    # Calculer les moyennes pour chaque groupe
    top_avgs = [top_df[f'{tag}_percentage'].mean() for tag in TAGS_OF_INTEREST]
    rest_avgs = [rest_df[f'{tag}_percentage'].mean() for tag in TAGS_OF_INTEREST]

    x = np.arange(len(TAGS_OF_INTEREST))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width / 2, top_avgs, width, label=f'Top (≥{threshold} comp.)', color='#FF6B6B',
                   edgecolor='black')
    bars2 = ax.bar(x + width / 2, rest_avgs, width, label=f'Reste (<{threshold} comp.)', color='#4ECDC4',
                   edgecolor='black')

    # Ajouter les valeurs
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Pourcentage moyen (%)', fontsize=12)
    ax.set_title('Spécialisation : Top Compétiteurs vs Reste', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(TAGS_OF_INTEREST)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('../../data/top_vs_rest.png', dpi=300, bbox_inches='tight')
    logger.success("Comparaison sauvegardée : ../../data/top_vs_rest.png")
    plt.close()


def generate_summary_stats(df):
    """Génère un résumé statistique."""
    logger.info("\n" + "=" * 60)
    logger.info("RÉSUMÉ STATISTIQUE DE LA SPÉCIALISATION")
    logger.info("=" * 60)

    for tag in TAGS_OF_INTEREST:
        percentage_col = f'{tag}_percentage'
        logger.info(f"\n{tag}:")
        logger.info(f"  Moyenne : {df[percentage_col].mean():.2f}%")
        logger.info(f"  Médiane : {df[percentage_col].median():.2f}%")
        logger.info(f"  Écart-type : {df[percentage_col].std():.2f}%")
        logger.info(f"  Min : {df[percentage_col].min():.2f}%")
        logger.info(f"  Max : {df[percentage_col].max():.2f}%")

        # Compétiteurs 100% spécialisés
        specialists = df[df[percentage_col] == 100.0]
        logger.info(f"  Compétiteurs 100% spécialisés : {len(specialists)}")

    logger.info("\n" + "=" * 60)
    logger.info(f"Total de compétiteurs analysés : {len(df)}")
    logger.info(f"Nombre moyen de participations : {df['total_competitions'].mean():.2f}")
    logger.info(f"Nombre médian de participations : {df['total_competitions'].median():.0f}")
    logger.info("=" * 60 + "\n")


def main():
    # Charger les données
    df = load_data()

    # Générer les statistiques
    generate_summary_stats(df)

    # Créer les visualisations
    plot_top_competitors_heatmap(df, top_n=30)
    plot_specialization_distribution(df)
    plot_average_specialization_bars(df)
    plot_top_vs_rest(df, threshold=5)

    logger.success("\n✅ Toutes les visualisations ont été générées avec succès !")
    logger.info("Fichiers créés dans le dossier data/ :")
    logger.info("  - heatmap_specialization.png")
    logger.info("  - distribution_specialization.png")
    logger.info("  - average_specialization.png")
    logger.info("  - top_vs_rest.png")


if __name__ == "__main__":
    main()
