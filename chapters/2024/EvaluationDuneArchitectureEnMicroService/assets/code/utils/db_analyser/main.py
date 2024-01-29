from github import Github, Repository
from src.db_analyser_naive import DB_Analyser_Naive
from src.db_analyser_code import DB_Analyser_Code

#repo_name = "amigoscode/microservices"
repo_name = "Theophile-Yvars/Bank_microservice"
access_token = ''

def naive_analyse(repository):
    analyser = DB_Analyser_Naive()
    return analyser.run(repository)

def code_analse(repository):
    analyser = DB_Analyser_Code(access_token)
    return analyser.run(repository)

if __name__ == '__main__':
    print("DOCKER COMPOSE ANALYSIS", flush=True)
    g = Github(access_token)
    repository: Repository = g.get_repo(repo_name)

    if naive_analyse(repository):
        print("\nNaive analyse OK 🎉\n")
    elif code_analse(repository):
        print("\nCode analyse OK 🎉\n")
    else:
        print("\nDB analyse KO .... 🔥\n")

    g.close()