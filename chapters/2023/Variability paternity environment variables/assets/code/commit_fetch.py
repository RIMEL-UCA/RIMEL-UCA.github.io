import json
import subprocess
from git import Repo
from git import Actor
import re
import glob
import os
from typing import List

import git
repository = Repo.init('.')

repository = Repo.init('./')

# print(repository.git.log("--patch", "8d9fd97773fa90a568024151126abe1399a48c7d"))
# print(repository.git.show("8d9fd97773fa90a568024151126abe1399a48c7d"))

# repository.git.log()

commits = dict()

# commits = {
#       commit_id: {
#           author: string
#           date: string
#           addition: [{ENV_VARIABLE: value},]
#           deletion: [{ENV_VARIABLE: value},]
#       }
#
#
# }

# git show ed5927e45d0a5cb60251b84e7a1faa5ed1a328b5
# Names of all the commits that modify environnment variables
names = []
for commit in repository.iter_commits():
    commit_name = commit.__str__()

    for commit_file in commit.stats.files.keys():

        if '.env' in commit_file or 'docker-compose' in commit_file:

            try:
                result = repository.git.log(
                    "--patch", f"{commit}", commit_file)
            except:
                # print("Cannot read the commit file ", commit_file, " in the commit ", commit_name)
                print("File ", commit_file,
                      " was deleted in the commit ", commit_name)
                continue
            names.append(commit_name)
            for line in result.splitlines():
                if len(line) > 0:
                    if line[0] == '+' or line[0] == '-':

                        commits[commit_name] = {}
                        commits[commit_name]['author'] = commit.author
                        commits[commit_name]['date'] = commit.committed_date
                        commits[commit_name]['addition'] = []
                        commits[commit_name]['deletion'] = []

                        all_env_variables_in_the_line = re.findall(
                            "(^[A-Z0-9_]+)(\=)(.*\n(?=[A-Z])|.*$)", line[1:])
                        for reg in all_env_variables_in_the_line:
                            # ('APP_PORT', '=', '3000')
                            (name, eq, value) = reg
                            if line[0] == '+':
                                commits[commit_name]['addition'].append(
                                    {name: {'file': commit_file, 'value': value}})
                            if line[0] == '-':
                                commits[commit_name]['deletion'].append(
                                    {name: {'file': commit_file, 'value': value}})

print(commits)

# Get Paternity users
paternals = {}
for name in names:
    #We get the attributes names coz 1 user can use different email address but the same Username
    author = commits[name]['author'].name
    if author in paternals.keys():
        paternals[author] = paternals[author] + 1
    else:
        paternals[author] = 1

sorted_paternals = {k: v for k, v in sorted(paternals.items(), key=lambda item: item[1], reverse=True)}
print(sorted_paternals)



WORKING_REPOSITORY: str = "."


def flatten(A):
    rt = []
    for i in A:
        if isinstance(i, list):
            rt.extend(flatten(i))
        else:
            rt.append(i)
    return rt


def extract_environment_variable(word: str) -> str:
    """
    >>> extract_environment_variable("Bonjour")
    >>> ''
    >>> extract_environment_variable("${MY_VARIABLE_ENV}aaa")
    >>> 'MY_VARIABLE_ENV'
    :param word: the word you want to check
    :return: '' if the word is not an envionment variable, else the filtered environment variable
    """

    temp_word: str = word

    # Remove the prefix in the word
    for letter in temp_word:
        if letter.isupper():
            break
        word = word[1:]

    #print(temp_word)

    temp_word = word
    # Remove the suffix in the word
    for letter in temp_word[::-1]:
        if letter.isupper():
            break
        word = word[:-1]

    filtered_regex = flatten(re.findall(r"(^[A-Z0-9_]+)", word))
    if len(filtered_regex) == 0 or len(filtered_regex[0]) <= 2:
        return ''

    if len(filtered_regex[0]) == len(word):
        return word


    return ''


def recover_environment_variable_in_a_file(url_file: str) -> List[str]:
    result = []
    with open(url_file) as file:
        for line_number, line in enumerate(file, 1):
            words_in_line = line.split()

            for word in words_in_line:
                env_variable = extract_environment_variable(word)
                if env_variable != '':
                    result.append({"line_number": line_number, "env_variable": env_variable})

    return result




types = ('yml', 'java') # the tuple of file types
files = []
for file_type in types:
    files.extend(glob.glob(f'{WORKING_REPOSITORY}/**/*.{file_type}', recursive=True))

files = [f for f in files if os.path.isfile(f)]

result = {}
for file_url in files:
    env_var = recover_environment_variable_in_a_file(file_url)
    if len(env_var) > 0:
        print("file", file_url, ":\n")

        for line in env_var:
            print("\t", line)
            line = str(line["line_number"])

            git_blame = subprocess.check_output(["git", "blame", file_url, "-L", line + "," + line, "--incremental"]).decode("utf-8").split("\n")
            print("\t\t", git_blame[1])
            print("\t\t", git_blame[2], "\n")
        print("\n")



#print(result)
