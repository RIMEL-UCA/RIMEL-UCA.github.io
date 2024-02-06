import requests
import os
from .utils_scraping import getTokenHeader

def get_comments_request(type, repo, headers, page=1, per_page=100):
    url = f"https://api.github.com/repos/{repo}/{type}comments?per_page={per_page}&page={page}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to fetch comments")
        return []

def get_comments(type, all_comments, repo, headers, page, file, all_file, number_of_comments):
    tmp_comments = []
    while len(all_comments) < int(number_of_comments):
        comments = get_comments_request(type, repo, headers, page=page)
        if not comments:
            break
        tmp_comments.extend(comments)
        all_comments.extend(comments)
        print(f"Comments fetched: {len(all_comments)}")
        page += 1
        if(len(all_comments) % 100000 == 0):
            write_comments_to_file(tmp_comments, file)

    write_comments_to_file(tmp_comments, file)
    write_comments_to_file(tmp_comments, all_file)

def write_comments_to_file(comments, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'a', encoding='utf-8') as f:
        for comment in comments:
            author_info = comment['user']
            message = comment['body']
            if author_info is not None:
                f.write(f"Author: {author_info['login']}, Date: {comment['created_at']}\n")
                f.write(f"Message: {message}\n")
                f.write("\n")

# Fonction principale pour scraper les données
def scrape_data():
    headers = getTokenHeader()
    repo = input("Enter GitHub repository (format: user/repo)(default: freeCodeCamp/freeCodeCamp): ")
    if repo == '':
        repo = 'freeCodeCamp/freeCodeCamp'
    all_comments = []
    commit_comments_file = f"results/{repo.replace('/', '_')}/comments/commit_comments.txt"
    issue_comments_file = f"results/{repo.replace('/', '_')}/comments/issue_comments.txt"
    pull_request_comments_file = f"results/{repo.replace('/', '_')}/comments/pull_request_comments.txt"
    final_comments_file = f"results/{repo.replace('/', '_')}/comments/all_comments.txt"
    page = 1
    number_of_comments = input("How many comments do you want to fetch? (100 min): ")
    if number_of_comments == '':
        number_of_comments = 2**31 -1
    print("Fetching comments...")
    get_comments('', all_comments, repo, headers, page, commit_comments_file, final_comments_file, number_of_comments)
    get_comments('issues/', all_comments, repo, headers, page, issue_comments_file, final_comments_file, number_of_comments)
    get_comments('pulls/', all_comments, repo, headers, page, pull_request_comments_file, final_comments_file, number_of_comments)
    print(f"Total comments fetched: {len(all_comments)}")
