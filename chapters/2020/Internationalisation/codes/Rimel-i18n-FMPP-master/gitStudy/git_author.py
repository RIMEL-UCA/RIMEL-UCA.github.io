# Source https://gist.github.com/simonw/091b765a071d1558464371042db3b959
# Modified by prune.pillone@etu.unice.fr

import subprocess
import re
import os

leading_4_spaces = re.compile('^    ')

keywords = ["localization", "l10n", "i18n", "internationalization", "translate", "translation", "weblate", "Weblate"]


def getName():
    list = os.getcwd().split("/")
    projectName = list[len(list) - 1]
    return projectName


def run():
    outputFile = open("./output/outputAuthor.txt", 'w')
    try:
        for f in os.walk("../Projects"):
            for folder in f[1]:
                os.chdir("../Projects/" + folder)
                print(getName())
                outputFile.write(getName() + "\n")

                commits = get_commits()

                authors = {}
                total = 0
                totalLoc = 0
                weblate = 0

                for commit in commits:
                    total += 1
                    found = 0
                    for word in keywords:
                        if found == 0 and (word in commit['message'] or word in commit['title']):
                            if ("Translated using Weblate" in commit['title']):
                                weblate += 1
                            else:
                                authorString = commit['author']
                                name = authorString.split('<')[0]
                                if name in authors.keys():
                                    authors[name] += 1
                                else:
                                    authors[name] = 1
                            totalLoc += 1
                            found = 1
                    authors["weblate"] = weblate
                for data in authors:
                    if (totalLoc != 0):
                        nbmCommit = authors[data]
                        authors[data] = (float(nbmCommit) / float(totalLoc)) * 100
                    else:
                        authors[data] = 0
                weblate2 = 0
                other = 0
                # treble shot = 42 weblate commit
                for dataPercentage in authors:
                    if dataPercentage != 'weblate':
                        other += authors[dataPercentage]
                outputFile.write("Weblate = " + str(authors['weblate']) + "\n")
                outputFile.write("Humains = " + str(other) + "\n")
                os.chdir("..")
        outputFile.close()
    except:
        print("over")


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
        print(file)
        number += 1
    outputFile.write(commit['hash'] + " = " + str(number) + "\n")


run()
