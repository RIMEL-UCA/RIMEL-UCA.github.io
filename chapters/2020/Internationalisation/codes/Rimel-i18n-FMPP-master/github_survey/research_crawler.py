import json
import requests
from time import time, sleep


def list_repositories():
    count = 0
    t = time()

    username = 'GregoirePeltier'
    content = requests.get("https://api.github.com/search/repositories?per_page=100&q=stars:>1000+language:Java&type=Repositories",  auth=(username,''))

    urls = set()
    while len(urls)<1000:
        j = json.loads(content.text)["items"]
        for i in range(len(j)):
            urls.add(j[i]['git_url'])
        try:
            url = list(filter(lambda x:"next" in x,content.headers["Link"].split(',')))[0]

            url =url.split(";")[0][1:-1].strip("<").strip(">")
            print("{} urls, next {}".format(len(urls),url))
            sleep(7)
            content = requests.get(url,auth=(username,''))
        except:
            print("error")

    return urls


print(list_repositories())
