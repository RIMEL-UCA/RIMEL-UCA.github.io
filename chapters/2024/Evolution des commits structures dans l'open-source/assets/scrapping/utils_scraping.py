import os
from dotenv import load_dotenv

# Fonction pour récupérer le header d'authentification
def getTokenHeader():
    load_dotenv()  # Charger les variables d'environnement
    github_token = os.getenv('GITHUB_TOKEN')
    return {
        'Authorization': 'token ' + github_token,
        'Accept': 'application/vnd.github.v3+json',
    }