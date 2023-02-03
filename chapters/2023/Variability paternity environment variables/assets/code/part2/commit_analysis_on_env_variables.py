from git import Repo
import re
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import numpy as np

repository = Repo.init('./spring-cloud-netflix')
commits = dict()

prefrix = "SPRING_CLOUD_NETFLIX"

for commit in tqdm(list(repository.iter_commits())):

    commit_name = commit.__str__()

    for commit_file in commit.stats.files.keys():

        if '.java' in commit_file:

            try:
                result = repository.git.log("--patch", f"{commit}", commit_file)
            except:
                # print("File ", commit_file, " was deleted in the commit ", commit_name)
                continue

            for line in result.splitlines():
                if len(line) > 0:
                    if line[0] == '+' or line[0] == '-':

                        string_to_search = ["System.getenv", "@Value(", "@Autowired"]
                        for word_to_search in string_to_search:
                            if word_to_search in line[1:]:

                                commits[commit_name] = {}
                                commits[commit_name]['author'] = str(commit.author)
                                commits[commit_name]['date'] = commit.committed_date
                                commits[commit_name]['addition'] = []
                                commits[commit_name]['deletion'] = []

                                if line[0] == '+':
                                    commits[commit_name]['addition'].append({commit_file: {'value': line[1:]}})
                                if line[0] == '-':
                                    commits[commit_name]['deletion'].append({commit_file: {'value': line[1:]}})


with open(prefrix + "commits.json", "w") as outfile:
    json.dump(commits, outfile)

authors = {}

for value in commits.values():
    if not value['author'] in authors:
        authors[value['author']] = {}
        authors[value['author']]['addition'] = 0
        authors[value['author']]['deletion'] = 0
        authors[value['author']]['contributions'] = 0

    authors[value['author']]['addition'] += len(value['addition'])
    authors[value['author']]['deletion'] += len(value['deletion'])

    authors[value['author']]['contributions'] += len(value['addition'])
    authors[value['author']]['contributions'] += len(value['deletion'])


total_contrib = sum([val['contributions'] for val in authors.values()])
total_add_contrib = sum([val['addition'] for val in authors.values()])
total_del_contrib = sum([val['deletion'] for val in authors.values()])
for person in authors:
    authors[person]['total_contribution_percentage'] = (authors[person]['contributions'] / total_contrib) * 100
    authors[person]['addition_percentage'] = (authors[person]['addition'] / total_contrib) * 100
    authors[person]['deletion_percentage'] = (authors[person]['deletion'] / total_contrib) * 100

with open(prefrix + "authors.json", "w") as outfile:
    json.dump(authors, outfile)

plt.pie([val['contributions'] for val in authors.values()], labels = [name for name in authors.keys()])
plt.savefig(prefrix + 'chart.png')
