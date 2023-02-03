import requests 

github_repos = set()

def fetch_repositories():
    url = 'https://api.github.com/search/repositories?q=accident+language:jupyter-notebook&per_page=100&page=1'
    response = requests.get(url)
    total_count = response.json()['total_count']

    for i in range(1, total_count//100):
        url = 'https://api.github.com/search/repositories?q=accident+language:jupyter-notebook&per_page=100&page=' + str(i)
        response = requests.get(url)
        try:
            data = response.json()['items']
            print("loading : " + str(len(github_repos)/total_count*100) + "%")
            print(len(github_repos))
            for item in data:
                github_repos.add(item['full_name'])
        except:
            break

    return github_repos