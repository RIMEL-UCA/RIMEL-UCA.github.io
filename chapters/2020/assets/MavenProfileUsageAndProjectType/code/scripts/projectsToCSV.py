import requests
import json
import time
import os
import csv
from dotenv import load_dotenv, find_dotenv

# Load companies file in an array
companies = []
f = open("../companies.in", "r")
lines = f.readlines()
for company in lines:
    companies.append(company)

# Languages to search for
languages = ['java', 'scala', 'kotlin']

# Load and find .env file
load_dotenv(find_dotenv())

# Get Github key from .env
GITHUB_KEY = os.environ.get("GITHUB_KEY")

# Store projects url to avoid duplications
urls = []

# Create out directory
if not os.path.exists('../out'):
    os.makedirs('../out')

# We search on the X first page of results
with open('../out/projects_from_companies_file.csv', mode='w') as projects_file:
    projects_writer = csv.writer(projects_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    projects_writer.writerow(['name', 'url', 'default_branch', 'stars'])

    # Browse companies 
    for company in companies:
        print('Switch to company ' + company)
        at_least_one = True
        for language in languages:
            nb = 1
            print('Switch to language ' + language)
            while(at_least_one):
                try:
                    # Request projects sort by stars
                    time.sleep(10)
                    # r = requests.get('https://api.github.com/search/repositories?q=language:maven pom&sort=stars&order=desc&page='
                    r = requests.get('https://api.github.com/search/repositories?q=user:' + company + '+language:' + language 
                    + '&sort=stars&order=desc&page=' 
                    + str(nb) + '&per_page=100', headers={'Authorization': 'Bearer ' 
                    + str(GITHUB_KEY)})

                    data = json.loads(r.text)

                    print('--> ' + str(len(data['items'])) + ' repositories retrieved on page ' + str(nb))

                    # Browse projects
                    for i in range (len(data['items'])):
                        # Write metadate in the file
                        url = data['items'][i]['html_url']
                        if url not in urls:
                            urls.append(url)
                            name = data['items'][i]['full_name']
                            branch = data['items'][i]['default_branch']
                            stars = data['items'][i]['stargazers_count']
                            projects_writer.writerow([name, url, branch, stars])
                    
                    nb = nb + 1
                    if len(data['items']) == 0:
                        at_least_one = False

                except requests.exceptions.RequestException as e:  # This is the correct syntax
                    print(e)
                    exit(1)

            print('--> Loop terminated for company ' + company + ' + language ' + language)
                
        time.sleep(10)