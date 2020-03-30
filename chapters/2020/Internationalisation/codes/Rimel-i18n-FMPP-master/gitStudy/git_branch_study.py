# Source https://gist.github.com/simonw/091b765a071d1558464371042db3b959
# Modified by prune.pillone@etu.unice.fr

import subprocess
import re
import os
import sys

leading_4_spaces = re.compile('^    ')

keywords = ["localization", "l10n", "i18n", "internationalization", "translate", "translation", "weblate"]


def getName():
    list = os.getcwd().split("/")
    # print(list[len(list) - 1])
    projectName = list[len(list) - 1]
    return projectName


def run():
    outputFile = open("./output/outputBranches.txt", 'w')
    for f in os.walk("../Projects"):
        for folder in f[1]:
            try:
                os.chdir("../Projects/" + folder)
                print(getName() + "\n")
                # Get all branches
                lines = subprocess.check_output(
                    ['git', 'branch', '--list', '-a'], stderr=subprocess.STDOUT
                ).split('\n')
                goodLines = []
                print("Branches")
                for line in lines:
                    line = line.replace(' ', '')
                    if "remotes/origin/HEAD" not in line and "*" not in line and line != "master":
                        goodLines.append(line)
                for branch in goodLines:
                    print(branch)
                    subprocess.check_output(
                        ['git', 'checkout', branch, "--force"], stderr=subprocess.STDOUT
                    )
                    #   Get commits
                    total = 0
                    totalLoc = 0
                    commits = get_commits()
                    for commit in commits:
                        total += 1
                        # print(commit['title'])  print(commit['message']) print(commit['hash'])
                        found = 0
                        for word in keywords:
                            if found == 0 and (word in commit['message'] or word in commit['title']):
                                totalLoc += 1
                                found = 1
                    outputFile.write(getName() + " " + branch + " = " + str(totalLoc) + "/" + str(total) + "\n")
                os.chdir("..")
            except:
                print("Over")
            exit(0)
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


run()
