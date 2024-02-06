import requests
import os
from .utils_scraping import getTokenHeader

# Fonction pour récupérer les commits d'un dépôt GitHub
def get_commits(repo, headers, page=1, per_page=100):
    commits_url = f"https://api.github.com/repos/{repo}/commits?per_page={per_page}&page={page}"
    response = requests.get(commits_url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to fetch commits")
        return []

# Fonction pour écrire les commits dans un fichier
def write_commits_to_file(commits, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'a', encoding='utf-8') as f:
        for commit in commits:
            author_info = commit['commit']['author']
            message = commit['commit']['message']
            f.write(f"Author: {author_info['name']}, Date: {author_info['date']}\n")
            f.write(f"Message: {message}\n")
            f.write("\n")
    

def remove_coauthored_and_extra_blank_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Filtrer les lignes qui ne commencent pas par "Co-authored-by"
    filtered_lines = [line for line in lines if not line.startswith('Co-authored-by')]

    # Supprimer les lignes vides consécutives
    final_lines = []
    previous_line_was_blank = False

    for line in filtered_lines:
        if line.strip() == '':
            if not previous_line_was_blank:
                final_lines.append(line)
                previous_line_was_blank = True
        else:
            final_lines.append(line)
            previous_line_was_blank = False

    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(final_lines)

def filter_file_lines(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        lines = file.readlines()

    # Filtrer les lignes qui commencent par "Author:", "Message:", ou sont des lignes vides
    filtered_lines = [line for line in lines if line.startswith('Author:') or line.startswith('Message:') or line == '\n']

    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(filtered_lines)

# Fonction principale pour scraper les données
def scrape_data():
    headers = getTokenHeader()

    repo = input("Enter GitHub repository (format: user/repo)(default: freeCodeCamp/freeCodeCamp): ")
    if repo == '':
        repo = 'freeCodeCamp/freeCodeCamp'
    all_commits = []
    commits_file = f"results/{repo.replace('/', '_')}/all_commits.txt"
    page = 1
    number_of_commits = input("How many commits do you want to fetch? (100 min): ")
    if number_of_commits == '':
        number_of_commits = 2**31 -1
    print("Fetching commits...")
    while len(all_commits) < int(number_of_commits):
        commits = get_commits(repo, headers, page=page)
        if not commits:
            break
        all_commits.extend(commits)
        print(f"Commits fetched: {len(all_commits)}")
        page += 1
        if(len(all_commits) % 100000 == 0):
            write_commits_to_file(all_commits, commits_file)

    write_commits_to_file(all_commits, commits_file)
    filter_file_lines(commits_file)
    remove_coauthored_and_extra_blank_lines(commits_file)
    
    print(f"Total commits fetched: {len(all_commits)}")
