import json
import os
import re
import urllib.request
from pathlib import Path
langs = {"en","fr","us","ru","it","es","pt"}
langroots = ["_{}.".format(lang) for lang in langs]

pattern = re.compile("^.+values(-[a-zA-Z]{1,3})+(/+\w{2,5})*/.+$")
def check_if_localised(path):
    stack = [path]
    while stack:
        path = stack.pop()
        for f in os.listdir(path) :
            fullPath = "{}/{}".format(path, f)
            if os.path.isdir(fullPath):
                if pattern.match(f):
                    return True
                stack.append(fullPath)
            else:
                if any(langroot in f for langroot in langroots):
                    return True
                if pattern.match(f):
                    return True
    return False
