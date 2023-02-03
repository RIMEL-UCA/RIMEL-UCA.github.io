from git import Repo
import re
from tqdm import tqdm
import json

repository = Repo.init('./Rocket.Chat')
commits = dict()

for commit in tqdm(list(repository.iter_commits())):

    commit_name = commit.__str__()

    for commit_file in commit.stats.files.keys():

        if 'docker-compose' in commit_file:

            commits[commit_name] = {}
            commits[commit_name]['author'] = str(commit.author)
            commits[commit_name]['date'] = commit.committed_date
            commits[commit_name]['addition'] = []
            commits[commit_name]['deletion'] = []

            try:
                result = repository.git.log("--patch", f"{commit}", commit_file)
            except:
                print("File ", commit_file, " was deleted in the commit ", commit_name)
                continue

            for line in result.splitlines():
                if len(line) > 0:
                    if line[0] == '+' or line[0] == '-':
                        all_env_variables_in_the_line = re.findall("^\s*-\s(\w+)=(.*)$", line[1:])

                        for reg in all_env_variables_in_the_line:
                            (name, value) = reg
                            if line[0] == '+':
                                commits[commit_name]['addition'].append({name: {'file': commit_file, 'value': value}})
                            if line[0] == '-':
                                commits[commit_name]['deletion'].append({name: {'file': commit_file, 'value': value}})


with open("commits.json", "w") as outfile:
    json.dump(commits, outfile)

authors = {}

for value in commits.values():
    if not value['author'] in authors:
        authors[value['author']] = {}
        authors[value['author']]['addition'] = 0
        authors[value['author']]['deletion'] = 0

    authors[value['author']]['addition'] += len(value['addition'])
    authors[value['author']]['deletion'] += len(value['deletion'])

with open("authors.json", "w") as outfile:
    json.dump(authors, outfile)
