from wsgiref import headers

import requests
import json
import numpy

# from github import Github

base_url = 'https://api.github.com'


def get_all_commits_count(owner, repo, sha):
    first_commit = get_first_commit(owner, repo)
    compare_url = '{}/repos/{}/{}/compare/{}...{}'.format(base_url, owner, repo, first_commit, sha)

    commit_req = requests.get(compare_url)
    commit_count = commit_req.json()['total_commits'] + 1
    print(commit_count)
    return commit_count


def get_first_commit(owner, repo):
    url = '{}/repos/{}/{}/commits'.format(base_url, owner, repo)
    print(url)
    req = requests.get(url)
    json_data = req.json()


    if req.headers.get('Link'):
        print(req.headers.get('Link'))
        page_url = req.headers.get('Link').split(',')[1].split(';')[0].split('<')[1].split('>')[0]
        print(page_url)
        req_last_commit = requests.get(page_url)
        first_commit = req_last_commit.json()
        #print(first_commit[-1])
        first_commit_hash = first_commit[-1]['sha']
    else:
        first_commit_hash = json_data[-1]['sha']
    return first_commit_hash


# Fonctionnalités à implémenter
# 0- Partir de l'hypothèse que le label "good first issue ou autre (feature, ...)" est un label par lequel un nouveau collaborateur entre dans le projet --> ok
# 1- Chercher les issues ayant pour label "good first issue ou autre" --> ok
# 2- recenser des auteurs de ces issues --> ok
# 3- Pour chaque auteur, recenser ses premiers commits --> ok
# 4- Analyser ses premiers commits et tirer une conclusion sur le code, le système en général, la partie traitée par l'auteur

# (1)
'''
retourne les premières issues dont le label est passé en paramètre
'''
def get_issues_by_labels(owner, repo, label):
    # array of issues
    issues = []
    # to know if the last page will really be 300
    last_page = 0


    url = '{}/repos/{}/{}/issues?labels={}'.format(base_url, owner, repo, label)
    print(url)
    req = requests.get(url)
    json_data = req.json()

    if req.headers.get('Link'):
        # print(req.headers.get('Link'))
        last_page_url = req.headers.get('Link').split(',')[1].split(';')[0].split('<')[1].split('>')[0]
        last_page_number = last_page_url.split('page=')[1]
        for page_num in range(1, int(last_page_number) + 1):
            print('page number : ' + str(page_num))
            try:
                url = '{}/repos/{}/{}/issues?labels={}&page={}'.format(base_url, owner, repo, label, page_num)
                # issue = requests.get(url).json()
                issues = issues + requests.get(url).json()
                # print(issues)
                last_page = page_num
            except:
                None
            break # Une seule fois pour 30 elements
    else:
        try:
            issues = issues + json_data
        except:
            None
    '''
    for page_num in range(1, 4):
        print(page_num)
        try:
            url = '{}/repos/{}/{}/issues?labels={}&page={}'.format(base_url, owner, repo, label, page_num)
            # issue = requests.get(url).json()
            issues = issues + requests.get(url).json()
            print(issues)
            last_page = page_num
        except:
            issues = issues + None
            # issues.append(None)
    '''
    # print('last page is ' + str(last_page))
    # print(last_page)
    print('echantillon de : ' + str(len(issues)) + ' issues de label "' + label + '"')
    # print(issues)
    return issues
    pass

# (2)
'''
retourne les auteurs des issues passés en paramètre ansi que leur date de création
'''
def get_issues_authors(issues):
    # print(len(issues))
    usersWithDates = []
    for issue in issues:
        try:
            userWithDate = {}
            userWithDate[issue['user']['login']] = issue['created_at']
            usersWithDates.append(userWithDate)
        except:
            None
    # on peut avoir plusieurs utilisateurs pour plusieurs issues, il faut donc les avoir de façon unique
    uniqueUsersWithDates = list(map(dict, set(tuple(sorted(sub.items())) for sub in usersWithDates)))
    return uniqueUsersWithDates
    pass

# (3)
'''
le lien entre les commits et les issues se font à partir des dates de création des issues en supposant que les auteurs committent dessus
'''
def get_commits_of_authors_issues(owner, repo, authors):
    # print('-----Adresses http-----')
    final_autors = 0
    all_commits = {}
    for author in authors:
        # array of commits
        # print(list(author.keys())[0])
        commits = []
        # last_page = 0
        # récupère les 30 premiers commits à partir de la date de céation de l'issue
        url = '{}/repos/{}/{}/commits?author={}&since={}'.format(base_url, owner, repo, list(author.keys())[0], list(author.values())[0])
        # print(url)
        req = requests.get(url)
        json_data = req.json()
        # print(req.headers.get('Link'))
        if req.headers.get('Link'):
            # print(req.headers.get('Link'))
            last_page_url = req.headers.get('Link').split(',')[1].split(';')[0].split('<')[1].split('>')[0]
            # print(last_page_url)
            req_last_commit = requests.get(last_page_url)
            commits = req_last_commit.json()
        else:
            try:
                commits = json_data
            except:
                None

        '''
        for page_num in range(1, 0, -1):
            try:
                url = '{}/repos/{}/{}/commits?author={}&since={}&page={}'.format(base_url, owner, repo, list(author.keys())[0], list(author.values())[0], page_num)
                print(url)
                commits = commits + requests.get(url).json()
                print(commits)
                break
                # last_page = page_num
            except:
                commits.append(None)
        '''
        # print(len(commits))
        # print(commits)
        if ( len(commits) != 0): final_autors = final_autors + 1
        all_commits[list(author.keys())[0]] = commits
        # break
    # print('last page is ' + str(last_page))
    # print(last_page)
    # print(len(all_commits))
    # print(all_commits)
    print('Résultats : ' + str(final_autors))
    '''for author in all_commits:
        print(author)
        print(all_commits[author])
    '''
    return all_commits
    pass

    '''for page in issues:
        for issue in page:
            goodFirstIssues = list(filter(lambda x: x['name'] == 'good first contribution', issues))
            try:
                all_repo_names.append(repo['full_name'].split("/")[1])
            except:
                pass'''

    # url = '{}/repos/{}/{}/labels'.format(base_url, owner, repo)
    # req = requests.get(url)
    # json_data = req.json()
    # while 'next' in issues.links.keys():
    #    req = requests.get(req.links['next']['url'])
    #    json_data.extend(req.json())
    # good = list(filter(lambda x: x['name'] == 'good first contribution', json_data))
    # https://www.getpostman.com/collections/436fe05eb3584999b52a
    # print(req.links)
    # print(json_data)


'''
while 'next' in res.links.keys():
  res=requests.get(res.links['next']['url'],headers={"Authorization": git_token})
  repos.extend(res.json())
'''


def main():
    # owners = ['flutter']
    # repos = [{'flutter': 'flutter'}, {'microsoft': 'vscode'}, {'facebook': 'react-native'}, {'kubernetes': 'kubernetes'}, {'tensorflow': 'tensorflow'}]
    repos = [{'tensorflow': 'tensorflow'}]
    # labels = ['good first contribution', 'engine', 'feature-request', 'bug']
    labels = ['good first contribution']
    author = 'jonahwilliams'
    # Took the last commit, Can do it automatically also but keeping it open
    sha = '0cd0c7ccace63de9a79dcc451d11b56dab2e5cca'
    # get_all_commits_count(owner, repo, sha)
    # print(get_first_commit(owner, repo))
    for repo in repos:
        owner = list(repo.keys())[0]
        repo_name = list(repo.values())[0]
        print("####################################################################################################################\n" + owner + "/" + repo_name + "\n####################################################################################################################\n")
        for label in labels:
            print(">>>>>>>>>>>>>>>>>>>>>> Label " + label + " <<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
            get_commits_of_authors_issues(owner, repo_name, get_issues_authors(get_issues_by_labels(owner, repo_name, label)))
            break
        break
    # get_latest_commits_by_author(owner, repo, author)


if __name__ == '__main__':
    main()
