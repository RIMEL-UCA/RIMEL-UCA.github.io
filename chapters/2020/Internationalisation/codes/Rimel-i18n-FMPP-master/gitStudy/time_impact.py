import collections
import os
import re
import sys

from git.repo.base import Repo
from tqdm import tqdm
from pydriller import RepositoryMining
android_pattern = re.compile("^.+values(-[a-zA-Z]{1,3})+(/+\w{2,5})*/.+$")
properties_pattern = re.compile("^.+_\\w{2}.properties$")

def get_changes(repo):
    paths = dict()
    changes = collections.defaultdict(lambda: set())
    for commit in tqdm(RepositoryMining(repo,clone_repo_to="../Project").traverse_commits()):
        for file in commit.modifications:

            path = file.old_path if file.old_path else file.new_path
            if android_pattern.match(path) or properties_pattern.match(path):
                changes[path].add(commit.committer_date)
                if file.old_path and file.new_path and file.old_path != file.new_path:
                    paths[file.new_path] = file.old_path
    stack = list(paths.keys())
    origins = set()
    while stack:
        p = stack.pop()
        t = paths[p]
        marked = set()
        while t in paths and t not in origins and t not in marked:
            marked.add(t)
            t = paths[t]
        origins.add(t)
        changes[t].update(changes[p])
        del(changes[p])
    return changes
with open("output/edition_dates.txt", 'w') as results:
    repositories = list(sys.stdin)
    for repo in tqdm(repositories,desc="Repositories : "):
        repo = repo.strip()
        name = repo.split("/")[-1].split(".")[0]
        path = "../Projects/{}"
        if not (os.path.isdir(path.format(name))):
            Repo.clone_from(repo, path.format(name))
        changes = get_changes(path.format(name))
        results.write("repository : ")
        results.write(repo+"\n")
        for path in changes:

            results.write(path+"\n")

            for date in changes[path]:
                results.write(str(date)+"\n")
            results.write("\n")
