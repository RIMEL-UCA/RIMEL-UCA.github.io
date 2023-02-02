'''
To create a json file with the repositories that contain the notebooks with the keyword of oop "class"
READ BEFORE RUNNING THIS CODE
'''

from github import Github
import json
from main import fetch_repositories

g = Github("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")

'''dict of noteebooks in the repository'''
repository_notebooks = {}

'''
notebook cells
'''
def get_notebook_cells(file_content):
    notebook_cells = {}
    notebook_cells[file_content.name] = []
    try:
        notebook_cells_content = json.loads(file_content.decoded_content.decode("utf-8"))['cells']
        for cell in notebook_cells_content:
            notebook_cells[file_content.name].append(cell['source'])
    except:
        pass
    return notebook_cells

'''
extract notebooks from repositories and store them in a dict "repository_notebooks"
'''
def fetch_notebooks(repository):
    try:
        repo = g.get_repo(repository)
    except:
        print("Error: " + repository)
    repository_notebooks[repo.full_name] = []
    contents = repo.get_contents("")
    while contents:
        file_content = contents.pop(0)
        if file_content.type == "dir":
            contents.extend(repo.get_contents(file_content.path))
        else:
            if file_content.path.endswith('.ipynb'):
                repository_notebooks[repo.full_name].append(get_notebook_cells(file_content))

'''max number of repos to fetch '''
number_max_of_repositories = 300

repositories = fetch_repositories()
for repository in repositories:
    if number_max_of_repositories == 0:
        break
    print(repository)
    fetch_notebooks(repository)
    number_max_of_repositories -= 1

#rename the json file before running this code
with open('notebooks1.txt', 'w') as f:
    f.write(json.dumps(repository_notebooks))
