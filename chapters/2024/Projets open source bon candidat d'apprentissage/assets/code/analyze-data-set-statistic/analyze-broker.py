from github import Github
import os
import yaml

def get_repo_list(file_path):
    with open(file_path, 'r') as file:
        repos = [line.strip() for line in file.readlines() if line.strip()]
    return repos

def get_docker_compose_images(repo):
    g = Github('Mon token')  # Remplacez os.getenv('GITHUB_ACCESS_TOKEN') par votre token d'accès GitHub

    try:
        repository = g.get_repo(repo)
        contents = repository.get_contents("")
        docker_compose_content = get_docker_compose_recursively(contents)
        return analyze_docker_compose(repo, docker_compose_content)
    except Exception as e:
        print(f"No docker-compose.yml found for {repo}: {e}")
        return None

def get_docker_compose_recursively(contents):
    for content_file in contents:
        if content_file.type == "dir":
            try:
                subdir_contents = content_file.repository.get_contents(content_file.path)
                result = get_docker_compose_recursively(subdir_contents)
                if result:
                    return result
            except Exception as e:
                print(f"Error accessing directory {content_file.path}: {e}")
        elif content_file.name == "docker-compose.yml":
            return content_file.repository.get_contents("docker-compose.yml").decoded_content.decode('utf-8')

def analyze_docker_compose(repo, content):
    try:
        docker_compose = yaml.safe_load(content)
        services = docker_compose.get('services', {})
        stats = {}

        for service_name, service_config in services.items():
            if 'kafka' in service_config.get('image', '').lower() or 'kafka' in service_config.get('broker', '').lower():
                image_name = service_config.get('image', '') or service_config.get('broker', '')
                stats[image_name] = stats.get(image_name, 0) + 1

        return stats
    except yaml.YAMLError as e:
        print(f"Error parsing docker-compose.yml for {repo}: {e}")
        return None

def main():
    repo_list_file = 'suite_a_stats.txt'  # Remplacez par le chemin de votre fichier contenant la liste des repos
    repositories = get_repo_list(repo_list_file)
    all_stats = {}

    for repo in repositories:
        print(f"Analyzing {repo}...")
        stats = get_docker_compose_images(repo)
        if stats:
            print(f"Statistics for {repo}: {stats}")
            for image, count in stats.items():
                all_stats[image] = all_stats.get(image, 0) + count
        else:
            print(f"No docker-compose.yml found for {repo}")

    save_statistics(all_stats)

def save_statistics(all_stats):
    stats_file = 'all_repos_statistics.txt'
    with open(stats_file, 'w') as file:
        file.write("Statistics for all repositories:\n")
        for image, count in all_stats.items():
            file.write(f"{image}: {count}\n")
        print(f"Statistics saved in {stats_file}")

if __name__ == "__main__":
    main()
