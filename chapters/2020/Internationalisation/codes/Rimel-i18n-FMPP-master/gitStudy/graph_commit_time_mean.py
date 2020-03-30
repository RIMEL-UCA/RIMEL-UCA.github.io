import datetime
import re
from collections import defaultdict
from math import floor
from statistics import mean

from matplotlib import pyplot as plt
from tqdm import tqdm


def is_date(text: str):
    try:
        parseDate(text)
        return True
    except ValueError as e:
        return False


def parseDate(text):
    try:
        text = text.strip()
        text = text.split("+")

        date_text = text[0] + "+" + text[1].replace(":", "")
    except IndexError:
        raise ValueError()
    date = datetime.datetime.strptime(date_text, "%Y-%m-%d %H:%M:%S%z")
    return date


def parse():
    repositories = dict()
    with open("./output/edition_dates.txt") as dates_file:
        currentRepo = dict()
        currentFile = None
        for line in dates_file:
            line = line.strip()
            if re.match(r"^repository : ", line):
                repositories[line] = currentRepo
            elif is_date(line):
                currentFile.add(parseDate(line))
            elif line != "\n":
                currentFile = set()
                currentRepo[line] = currentFile
    return repositories


def countDatePerKey(files, key):
    counts = defaultdict(lambda: 0)
    for file in files:
        for day in files[file]:
            counts[key(day)] += 1
    return counts

def normalizeDateCount(counts):
    start = min(map(lambda x:x.timestamp(),counts))
    end = max(map(lambda x:x.timestamp(),counts))
    normalized = dict()
    for d in counts:
        ts = d.timestamp()
        percent = floor((ts - start) / (end-start) * 100)
        normalized[percent] = counts[d]
    return normalized

repositories = parse()
counts = dict()
counts = []
for rep in tqdm(repositories,desc="Counting"):

    rep_counts = countDatePerKey(repositories[rep] ,
                             lambda x: x.replace(second=0, microsecond=0, minute=0, hour=0))
    rep_counts = normalizeDateCount(rep_counts)
    counts.append(rep_counts)
mean_counts = dict()
for i in tqdm(range(101)):
    values = []

    for c in counts:
        if i in c:
            values.append(c[i])
    mean_counts[i] = mean(values) if values else 0

# counts = {key:counts[key] for key in tqdm(counts,desc="Filtering") if counts[key]<1000}
counts = mean_counts
plt.title("Nombre moyen de fichiers commit lié a la localisation dans le temps ")
plt.xlabel("Pourcentage de temps depuis le début de la localisation")
plt.ylabel("Nombre moyen de fichiers commits par pourcent")
plt.bar(counts.keys(),counts.values())
plt.savefig("moyenne_commit_lie_a_la_localisation_dans_le_temps")
plt.show()

