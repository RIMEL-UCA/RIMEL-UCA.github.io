from github import Github, Repository
from dockerfile_parse import DockerfileParser
import yaml

from utils.Colors import Couleurs


class Docker_compose_analyser():
    def __init__(self):
        print("Info ", flush=True)

    def run(self, repository):
        print("DOCKER COMPOSE PRESENCE DETECTION", flush=True)
        return self.__has_docker_compose(repository)

    def __has_docker_compose(self, repository: Repository):
        contents = repository.get_contents("")
        for content in contents:
            if content.name.lower() == 'docker-compose.yml':
                print("[ "+Couleurs.VERT+"DOCKER COMPOSE FOUND"+Couleurs.RESET+" ]")
                return True
        print("[ " + Couleurs.ROUGE + "DOCKER COMPOSE NOT FOUND" + Couleurs.RESET + " ]")
        return False

    def get_all_directories(self, repository, path=""):
        directories = []

        def get_directories_recursive(repo, directory_path):
            nonlocal directories
            for content in repo.get_contents(directory_path):
                if content.type == "dir":
                    directories.append(content.path)
                    get_directories_recursive(repo, content.path)

        get_directories_recursive(repository, path)
        return directories

    def calculate_directory_sizes(self, repo, directories):
        directory_sizes = {}
        for directory in directories:
            contents = repo.get_contents(directory)
            size = sum(file.size for file in contents if file.type == "file")
            directory_sizes[directory] = size

        return directory_sizes

    def has_docker_compose(self, repository):
        def search_for_docker_compose(contents, current_path=""):
            dockerfile = 0
            for content in contents:
                #print("content : ", content)
                if content.type == "dir":
                    sub_contents = repository.get_contents(content.path)
                    path = current_path + "/" + content.name if current_path else content.name
                    docker_compose_path = search_for_docker_compose(sub_contents, path)
                    if docker_compose_path:
                        return docker_compose_path
                elif (
                        content.name.lower() == 'docker-compose.yml' or 'docker-compose' in content.name.lower()) and (
                        '.yaml' or '.yml' in content.name.lower()):
                    docker_compose_content = repository.get_contents(content.name).decoded_content.decode("utf-8")
                    return docker_compose_content  # Retourne le contenu du fichier 'docker-compose.yml'
                elif content.name.lower() == 'dockerfile':
                    dockerfile += 1
            return None

        contents = repository.get_contents("")
        return search_for_docker_compose(contents)

    def get_services_from_docker_compose(self, repository, dockercompose):
        if dockercompose is None:
            return None
        return self.get_docker_services_from_compose(dockercompose)

    def get_docker_images_from_compose(self, file_content):
        images_list = []

        # Charger le contenu du fichier YAML
        compose_data = yaml.safe_load(file_content)

        # Vérifier si 'services' est présent dans le fichier docker-compose.yml
        if 'services' in compose_data:
            services = compose_data['services']

            # Parcourir chaque service pour obtenir les images
            for service_name, service_config in services.items():
                if 'image' in service_config:
                    images_list.append(service_config['image'])
                elif 'build' in service_config:
                    dockerfile_path = service_config['build']
                    if isinstance(dockerfile_path, str):  # Handle different build configurations
                        dockerfile_path = dockerfile_path.split(' ')[0]
                    dockerfile = DockerfileParser(path=dockerfile_path)
                    for instruction in dockerfile.structure:
                        if instruction['instruction'] == 'FROM':
                            images_list.append(instruction['value'])

        return images_list

    def get_docker_services_from_compose(self, file_content):
        service_names = []

        # Charger le contenu du fichier YAML
        compose_data = yaml.safe_load(file_content)

        # Vérifier si 'services' est présent dans le fichier docker-compose.yml
        if 'services' in compose_data:
            services = compose_data['services']

            # Obtenir les noms des services
            service_names = list(services.keys())

        return service_names

