import random
import webbrowser
from collections import defaultdict

from tqdm import tqdm

commits = defaultdict(lambda :set())
with open("output_confiance.txt") as selected:
    for line in selected :
        line = line.split()
        commits[line[0]].add(line[1])
commitCount = 10
chrome = webbrowser.get(using='google-chrome')
while commitCount:
    repo = random.choice(list(commits.keys()))
    print(repo)
    print("root url ? ( https://github.com/author/.../../commit/ ) ")
    root = input()
    if root =="n":
        continue
    safe = 0
    for commit in tqdm(commits[repo]):
        url = root+commit
        chrome.open(url, new=1)
        s = input()
        if s!="n":
            safe +=1
        commitCount-=1
    print("{}  safe : {} fiability : {}".format(repo,safe,safe/len(commits[repo])))
new=1


