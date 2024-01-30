from github import Github, Repository

from utils.db_analyser.src.db_analyser_naive import DB_Analyser_Naive
from utils.db_analyser.src.db_analyser_code import DB_Analyser_Code

class DB_analyser():
    def __init__(self, token):
        self.access_token = token
        print("DB analyser")

    def run(self, repository, dockercompose):
        print("DOCKER COMPOSE ANALYSIS", flush=True)

        #print("COMPOSE : ", dockercompose)
        print("Repo : ", repository, flush=True)

        is_naive_works, resutlat = self.__naive_analyse(dockercompose)
        res = {}
        if is_naive_works:
            print("\nNaive analyse OK :)\n", flush=True)
            res = resutlat

        if not is_naive_works:
            is_code_works, resutlat = self.__code_analse(repository, dockercompose)
            if is_code_works:
                print("\nCode analyse OK :)\n", flush=True)
                res = resutlat

        if len(resutlat) == 0: # pas de BD detecté
            print("\nNO DB : KO .... :(\n", flush=True)
            return -2
        elif all(valeur == 1 for valeur in resutlat.values()): # 1 db pour 1 service
            return 1
        elif all(valeur != 1 for valeur in resutlat.values()): # toutes les dbs sont connectées à plusieur service
            return 0
        else: # au moin une db est connecté à plusieur service
            return -1

    def __naive_analyse(self, dockercompose):
        analyser = DB_Analyser_Naive()
        return analyser.run(dockercompose)

    def __code_analse(self, repository, docker_compose_content):
        analyser = DB_Analyser_Code(self.access_token)
        return analyser.run(repository, docker_compose_content)
