#!/usr/bin/env python
# Imports
from typing import OrderedDict
import requests
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import datetime
import collections

# Code
class History():
    def __init__(self):
        self.history = dict()
    
    def incrementDate(self, date: str):
        date_string = date.split("T")[0]
        date_format = "%Y-%m-%d"
        try:
            date_obj = datetime.datetime.strptime(date_string, date_format)
        except ValueError:
            print("Incorrect data format, should be YYYY-MM-DD")

        if date_obj in self.history:
            self.history[date_obj] = self.history[date_obj] + 1
        else:
            self.history[date_obj] = 1

def analyzeResponse(history: History, commits: list):
    if commits:
        for commit in commits:
            history.incrementDate(commit["commit"]["committer"]["date"])

history = History()
page = 1
project = "CovidShield"
repository = "mobile"
r = requests.get("https://api.github.com/repos/{project}/{repository}/commits?per_page=100&page={page}".format(project=project, repository=repository, page=page))
commits: list = r.json()
analyzeResponse(history, commits)

while (commits):
    r = requests.get("https://api.github.com/repos/{project}/{repository}/commits?per_page=100&page={page}".format(project=project, repository=repository, page=page))
    commits: list = r.json()
    analyzeResponse(history, commits)
    page = page + 1

history_copy = history.history.copy()
history_sorted = sorted(history_copy.items())

previousDate : datetime.datetime = next(iter(history_sorted))
for date in history_sorted:
    duration : datetime.timedelta = date[0] - previousDate[0]
    if (duration.days - 1 > 0):
        for i in range(duration.days - 1):
            history.history[previousDate[0] + datetime.timedelta(days=i + 1)] = 0
    previousDate = date

chronologie = sorted(history.history)
od = collections.OrderedDict(sorted(history.history.items()))

commitsDate = [key.strftime("%d/%m/%Y") for key in od.keys()]
for i in range(len(commitsDate)):
    if i%10 != 0:
        commitsDate[i] = ""

commitsAmount = [value for value in od.values()]

y_pos = np.arange(len(commitsAmount))

plt.bar(y_pos, commitsAmount, align='center', alpha=0.5)
plt.xticks(y_pos, commitsDate)
plt.xlabel('Date')
plt.ylabel('Number of commits')

plt.show()
