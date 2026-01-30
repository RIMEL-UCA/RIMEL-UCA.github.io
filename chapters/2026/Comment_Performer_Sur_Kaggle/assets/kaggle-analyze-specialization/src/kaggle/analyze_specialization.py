#!/usr/bin/env python3
"""
Analyse de la spécialisation des top compétiteurs Kaggle
"""

from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
from loguru import logger
from collections import defaultdict
import time

TAGS_OF_INTEREST = ["NLP", "Time Series Analysis", "Tabular"]

NUM_PAGES = 25

TOP_N_LEADERBOARD = 50  # On prend le top 50 de chaque compétition

def get_filtered_competitions(api):
    """Récupère toutes les compétitions filtrées par tags."""
    logger.info("Récupération des compétitions")
    all_competitions = []

    for page in range(1, NUM_PAGES + 1):
        logger.debug(f"Récupération page {page}...")
        try:
            competitions = api.competitions_list(page=page)
            if not competitions:
                logger.warning(f"Aucune compétition trouvée à la page {page}")
                break
            all_competitions.extend(competitions.competitions)
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la page {page}: {e}")
            break

    logger.info(f"Total de compétitions récupérées : {len(all_competitions)}")

    # Filtrer par tags
    logger.info(f"Filtrage par tags : {TAGS_OF_INTEREST}")
    filtered_competitions = []

    for comp in all_competitions:
        if comp.tags:
            comp_tag_names = [tag.name for tag in comp.tags]

            # Vérifier si au moins un de nos tags est présent
            matched_tags = []
            for tag_comp in comp_tag_names:
                for tag_interest in TAGS_OF_INTEREST:
                    if tag_interest.lower() in tag_comp.lower():
                        matched_tags.append(tag_interest)
                        break

            if matched_tags:
                filtered_competitions.append({
                    'ref': comp.ref,
                    'title': comp.ref.split('/')[-1],
                    'tags': comp_tag_names,
                    'matched_tags': matched_tags,  # Tags qui matchent nos intérêts
                    'category': comp.category
                })

    logger.success(f"Compétitions filtrées : {len(filtered_competitions)}")
    return filtered_competitions


def get_leaderboard_for_competitions(api, competitions):
    """Récupèrent les leaderboards pour toutes les compétitions."""
    logger.info(f"Récupération des leaderboards pour {len(competitions)} compétitions")

    competitors_data = defaultdict(lambda: defaultdict(int))  # {username: {tag: count}}
    competition_errors = []

    for i, comp in enumerate(competitions):
        comp_title = comp['title']

        logger.info(f"[{i + 1}/{len(competitions)}] Récupération du leaderboard de : {comp_title}")

        try:
            # Récupérer le leaderboard (top TOP_N_LEADERBOARD)
            leaderboard = api.competition_leaderboard_view(
                competition=comp_title,
                page_size=TOP_N_LEADERBOARD
            )

            if leaderboard:
                logger.success(f"  → {len(leaderboard)} compétiteurs trouvés")

                # Pour chaque compétiteur du leaderboard
                for submission in leaderboard:
                    username = submission.team_name  # ou submission.teamName selon l'API

                    # Compter la participation dans chaque tag
                    for tag in comp['matched_tags']:
                        competitors_data[username][tag] += 1
            else:
                logger.warning(f"  → Aucun leaderboard disponible")

        except Exception as e:
            logger.error(f"  → Erreur : {e}")
            competition_errors.append({'competition': comp_title, 'error': str(e)})

        # Pause pour éviter de surcharger l'API
        time.sleep(0.5)

    if competition_errors:
        logger.warning(f"Erreurs rencontrées sur {len(competition_errors)} compétitions")

    return competitors_data


def build_specialization_matrix(competitors_data):
    """Construit la matrice de spécialisation."""
    logger.info("Construction de la matrice de spécialisation")

    # Convertir en DataFrame
    data = []
    for username, tag_counts in competitors_data.items():
        total_participations = sum(tag_counts.values())
        row = {
            'username': username,
            'total_competitions': total_participations
        }

        # Calculer les pourcentages pour chaque tag
        for tag in TAGS_OF_INTEREST:
            count = tag_counts.get(tag, 0)
            percentage = (count / total_participations * 100) if total_participations > 0 else 0
            row[f'{tag}_count'] = count
            row[f'{tag}_percentage'] = round(percentage, 2)

        data.append(row)

    df = pd.DataFrame(data)

    # Trier par nombre total de participations (pour avoir les top compétiteurs en premier)
    df = df.sort_values('total_competitions', ascending=False)

    return df


def main():
    # Initialiser l'API Kaggle
    logger.info("Initialisation de l'API Kaggle...")
    api = KaggleApi()
    api.authenticate()
    logger.success("Connexion à l'API Kaggle réussie !")

    # Étape 1 : Récupérer les compétitions filtrées
    filtered_competitions = get_filtered_competitions(api)

    if not filtered_competitions:
        logger.error("Aucune compétition trouvée avec les tags spécifiés")
        return

    # Sauvegarder la liste des compétitions
    competitions_df = pd.DataFrame(filtered_competitions)
    competitions_df.to_csv('../../data/filtered_competitions.csv', index=False)
    logger.success(f"Liste des compétitions sauvegardée dans data/filtered_competitions.csv")

    # Étape 2 : Récupérer les leaderboards
    competitors_data = get_leaderboard_for_competitions(api, filtered_competitions)

    logger.info(f"Total de compétiteurs uniques trouvés : {len(competitors_data)}")

    # Étape 3 : Construire la matrice de spécialisation
    specialization_matrix = build_specialization_matrix(competitors_data)

    # Sauvegarder la matrice
    specialization_matrix.to_csv('../../data/specialization_matrix.csv', index=False)
    logger.success(f"Matrice de spécialisation sauvegardée dans data/specialization_matrix.csv")

    # Afficher un aperçu des top compétiteurs
    logger.info("\n=== TOP 20 COMPÉTITEURS ===")
    print(specialization_matrix.head(20).to_string())

    # Statistiques globales
    logger.info("\n=== STATISTIQUES GLOBALES ===")
    for tag in TAGS_OF_INTEREST:
        avg_percentage = specialization_matrix[f'{tag}_percentage'].mean()
        logger.info(f"Moyenne de spécialisation en {tag} : {avg_percentage:.2f}%")


if __name__ == "__main__":
    main()
