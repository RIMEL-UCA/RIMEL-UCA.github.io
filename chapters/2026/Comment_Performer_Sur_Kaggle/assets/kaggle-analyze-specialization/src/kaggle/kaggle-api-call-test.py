import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
from loguru import logger

if __name__ == '__main__':
    logger.info("Kaggle API Client Started. Version {version}", version=kaggle.__version__)

    api = KaggleApi()
    api.authenticate()

    logger.info("Successfully authenticated to Kaggle API.")
    logger.info("Kaggle Configuration directory {config}", config=api.config)

    comp_iterator = api.competitions_list(search="")

    logger.info("Fetch 20 competitions")

    for competition in comp_iterator.competitions:
        logger.info("-" * 40)
        logger.info("Competition '{title}'", title=competition.title)
        logger.info("{{HostName: {name}, Category: {category}, tagCounts: {tagCounts}}}", name=competition.host_name,
                    category=competition.category,
                    tagCounts=len(competition.tags))
        logger.info("Tags:")
        for index, tag in enumerate(competition.tags):
            logger.info("   Tag n° {index} - {tag}", index=index, tag=tag.name.upper())

    leaderboard = api.competition_leaderboard_view(competition='titanic')

    for competitor in leaderboard:
        logger.info("Competitor - {{Name: {name}, Score: {score}}}", name=competitor.team_name, score=competitor.score)
