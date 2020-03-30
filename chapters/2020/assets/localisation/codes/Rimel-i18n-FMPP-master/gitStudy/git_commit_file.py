# Source https://gist.github.com/simonw/091b765a071d1558464371042db3b959
# Modified by prune.pillone@etu.unice.fr

## peut etre unterressant de voir qui commit

import subprocess
import re
import os
import sys

leading_4_spaces = re.compile('^    ')

keywords = ["localization", "l10n", "i18n", "internationalization", "translate", "translation", "weblate"]


def getName():
    list = os.getcwd().split("/")
    #print(list[len(list) - 1])
    projectName = list[len(list) - 1]
    return projectName


def run():
    outputFile = open("./output/outputFile.txt", 'w')
    for f in os.walk("../Projects"):
        for folder in f[1]:
            os.chdir("../Projects/" + folder)
            commits = get_commits()
            total = 0
            totalLoc = 0
            print(getName())
            outputFile.write(getName() + " = ")
            for commit in commits:
                found = 0
                for word in keywords:
                    if found == 0 and (word in commit['message'] or word in commit['title']):
                        get_Files(commit, outputFile)
                        found = 1
            outputFile.write("\n")
            os.chdir("..")
    outputFile.close()


def get_commits():
    lines = subprocess.check_output(
        ['git', 'log'], stderr=subprocess.STDOUT
    ).split('\n')
    commits = []
    current_commit = {}

    def save_current_commit():
        if 'message' in current_commit:
            title = current_commit['message'][0]
            message = current_commit['message'][1:]
            if message and message[0] == '':
                del message[0]
            current_commit['title'] = title
            current_commit['message'] = '\n'.join(message)
            commits.append(current_commit)

    for line in lines:
        if not line.startswith(' '):
            if line.startswith('commit '):
                if current_commit:
                    save_current_commit()
                    current_commit = {}
                current_commit['hash'] = line.split('commit ')[1]
            else:
                try:
                    key, value = line.split(':', 1)
                    current_commit[key.lower()] = value.strip()
                except ValueError:
                    pass
        else:
            current_commit.setdefault(
                'message', []
            ).append(leading_4_spaces.sub('', line))
    if current_commit:
        save_current_commit()
    return commits


def get_Files(commit, outputFile):
    number = 0
    files = subprocess.check_output(
        ['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', commit['hash']], stderr=subprocess.STDOUT).split(
        '\n')
    for file in files:
        number += 1
    outputFile.write(str(number) + ";")


run()
