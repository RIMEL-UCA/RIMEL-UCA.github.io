from github import Github

def test_kafka_docker_compose(fichier,repo):
    contenu_fichier = fichier.decoded_content.decode('utf-8')
    if 'kafka' in contenu_fichier:    
         with open('results_repo_kafka_csv.txt', 'a') as fichier_resultats:
            fichier_resultats.write(repo.full_name + '\n')
            return True
    else:   
        return False


def recherche_docker_compose_kafka(repo):
    try:
        # Récupérer la racine du référentiel
        root_contents = repo.get_contents("")
        for content_file in root_contents:
            # Vérifier si c'est un dossier
            if content_file.type == "dir":
                # Récupérer récursivement tous les fichiers du dossier
                for fichier in repo.get_contents(content_file.path):
                    if fichier.type == 'file' and fichier.name == 'docker-compose.yml':
                        test_kafka_docker_compose(fichier,repo)

            if content_file.type == 'file' and content_file.name == 'docker-compose.yml':
                test_kafka_docker_compose(content_file,repo)

    except Exception as e:
        print(f"Erreur lors de l'accès au référentiel {repo.full_name}: {str(e)}")
    return False

def recherche_repos_github():
    g = Github('Token')

    nombre_repos_a_rechercher = 1200
    nb_repos_trouves = 0

    repos = g.search_repositories(query=f"microservices kafka in:description")[:nombre_repos_a_rechercher]

    for repo in repos:
        print(f"Test Nom du dépôt : {repo.full_name}")
        if nb_repos_trouves >= nombre_repos_a_rechercher:
            break

        # Recherche récursive du fichier docker-compose.yml contenant Kafka
        if recherche_docker_compose_kafka(repo):
            nb_repos_trouves += 1

if __name__ == "__main__":
    recherche_repos_github()