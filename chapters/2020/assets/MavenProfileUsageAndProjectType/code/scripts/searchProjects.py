import requests
import json
import os
from time import sleep
from dotenv import load_dotenv, find_dotenv

# Load and find .env file
load_dotenv(find_dotenv())

# Get Github key from .env
GITHUB_KEY = os.environ.get("GITHUB_KEY")

# Keyword to look for
KEYWORD = "profile"

# Projects
projects = []
projects_ignored = []

for nb in range (1, 31):
    sleep(7)
    r = requests.get('https://api.github.com/search/repositories?q=language:Maven Pom&sort=stars&order=desc&page=' + str(nb))
    if r.status_code != 200:
        print('error on requesting projects')
        break
    data = json.loads(r.text)

    print(str(len(data['items'])), ' projects retrieved @ page ', nb)

    # Browse projects
    for i in range (0, len(data['items'])):
        # Check for every project if there is a profile
        url = 'https://api.github.com/search/code?q=' + KEYWORD + '+repo:' + data['items'][i]['full_name'] + '+in:file'
        
        # Please use your own Github token
        sleep(7)
        r2 = requests.get(url, headers={'Authorization': 'Bearer ' + str(GITHUB_KEY)})

        # If request is not ok
        if r2.status_code != 200:
            projects_ignored.append(data['items'][i]['html_url'])
            continue

        # Save if there is more than one call to kafka-node
        data_project = json.loads(r2.text)
        if int(data_project['total_count']) > 0:
            projects.append(data['items'][i]['html_url'])

print('-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_')
print(str(len(projects)), ' projects found with the key word ', KEYWORD)
print(projects)