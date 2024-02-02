from github import Github

def recherche_docker_compose(repo):
    docker_compose_found = False
    docker_compose_path = None

    contents = repo.get_contents("")


    while contents:
        content = contents.pop(0)
        print(content)

        if content.type == "file" and (content.name == 'docker-compose.yaml' or content.name == 'docker-compose.yml'):
            docker_compose_found = True
            docker_compose_path = content.path
            break

        if content.type == "dir":
            sub_contents = repo.get_contents(content.path)
            contents.extend(sub_contents)

    return docker_compose_found, docker_compose_path

def rechercher_depots(keywords, max_repos=100):
    g = Github("Token") 
    
    repos = g.search_repositories(query=f"{keywords} in:description", order='desc')[:max_repos]

    resultats = []

    for repo in repos:
        print(f"Test Nom du dépôt : {repo.full_name}")
        try:
            docker_compose_exists, docker_compose_path = recherche_docker_compose(repo)
  
            print(f"Docker Compose exist ? : {docker_compose_exists}")
            if docker_compose_exists:
                docker_compose_content = repo.get_contents(docker_compose_path).decoded_content.decode('utf-8')
                if 'kafka' in docker_compose_content:
                    lines = docker_compose_content.split('\n')
                    kafka_image = None
                    for line in lines:
                        if 'image' in line and 'kafka' in line:
                            kafka_image = line.split('image:')[-1].strip().replace("'", "").replace('"', '')
                            break

                        with open('resultats_depots_kafka.txt', 'w', encoding='utf-8') as fichier_resultats:
                            fichier_resultats.write(f"Nom du dépôt : {repo.full_name}\n")
                            fichier_resultats.write(f"Description : {repo.description}\n")
                            fichier_resultats.write(f"URL : {repo.html_url}\n")
                            fichier_resultats.write(f"Image Kafka : {kafka_image}\n")
                            fichier_resultats.write("\n")

        except Exception as e:
            print(f"Erreur lors de l'accès au dépôt {repo.full_name}: {e}")
            continue

    return resultats

# Exemple de recherche de dépôts GitHub avec des mots-clés spécifiques et vérification des fichiers Docker Compose
resultats = rechercher_depots('microservices', max_repos=100)

# Écriture des résultats dans un fichier texte
with open('resultats_depots_kafka.txt', 'w', encoding='utf-8') as fichier_resultats:
    for repo in resultats:
        fichier_resultats.write(f"Nom du dépôt : {repo['nom']}\n")
        fichier_resultats.write(f"Description : {repo['description']}\n")
        fichier_resultats.write(f"URL : {repo['url']}\n")
        fichier_resultats.write(f"Image Kafka : {repo.get('kafka_image', 'Non trouvée')}\n")
        fichier_resultats.write("\n")
