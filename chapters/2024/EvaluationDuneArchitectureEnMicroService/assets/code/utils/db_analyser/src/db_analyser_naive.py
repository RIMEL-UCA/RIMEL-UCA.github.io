import yaml

from utils.db_analyser.src.csv_manager import CSV_Manager
from utils.Colors import Couleurs

class DB_Analyser_Naive():
    def __init__(self):
        print("BD ANALYSER", flush=True)
        self.__csv_manager = CSV_Manager()

    def run(self, docker_compose_content):
        if docker_compose_content:
            compose_data = self.__load_docker_compose(docker_compose_content)

            if compose_data:
                db_usage_count = self.__count_db_usage_naive(compose_data)
                self.__check_single_usage(db_usage_count)


                # Vérifier si l'un des éléments est égal à 0
                if 0 in db_usage_count.values():
                    print("\n[ "+Couleurs.ROUGE + "KO" + Couleurs.RESET+" ] Au moins une DB n'est pas dans les depends_on ....\n")
                    return False, {}

                if len(db_usage_count) > 0:
                    print("\n[ " + Couleurs.VERT + "OK" + Couleurs.RESET + " ] Depends_on trouvé \n")
                    return True, db_usage_count
                else:
                    print("\n[ "+Couleurs.ROUGE + "KO" + Couleurs.RESET+" ] Pas de depends_on trouvé  ...\n")
                    return False, {}
            else:
                return False, {}

        return False, {}

    def __load_docker_compose(self, file_content):
        compose_data = yaml.safe_load(file_content)
        return compose_data

    def __is_database_service(self, service_name, image_name):
        keywords_for_db = ['mongo', 'mysql', 'postgres', 'cassandra']
        for keyword in keywords_for_db:
            if keyword in image_name.lower():
                print("DB detected ... : ", image_name)
                return True
        return False

    def __count_db_usage_naive(self, compose_data):
        db_names_in_services = set()
        db_usage_count = {}
        for service_name, service_config in compose_data.get('services', {}).items():
            image_name = service_config.get('image', '')
            if self.__is_database_service(service_name, image_name):
                db_names_in_services.add(service_name.lower())



        for service_name, service_config in compose_data.get('services', {}).items():
            depends_on = service_config.get('depends_on', [])
            for dependency_name in depends_on:
                db_name = dependency_name.lower()
                #print("NAME : ", db_name)
                if db_name in db_names_in_services:
                    db_usage_count[db_name] = db_usage_count.get(db_name, 0) + 1

        print(db_usage_count)

        return db_usage_count

    def __check_single_usage(self, db_usage_count):
        self.__csv_manager.write(db_usage_count)
        for db_name, usage_count in db_usage_count.items():
            if usage_count > 1:
                print("["+Couleurs.JAUNE+" WARNING "+Couleurs.RESET+"]: Database "+Couleurs.VERT+""+db_name+""+Couleurs.RESET+" is used by "+Couleurs.ROUGE+" ",usage_count,""+Couleurs.RESET+" services. It should be used by at most one service. KO")
            else:
                print(f"Database '{db_name}' is used by {usage_count} services. OK")