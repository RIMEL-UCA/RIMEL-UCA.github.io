from github import Github, Repository

class event_analyser():
    def __init__(self):
        print("EVENTS ANALYZER")
    def run(self, repo_name):
        g = Github(self.__access_token)
        repository: Repository = g.get_repo(repo_name)
        return self.__check_event_sourcing()

    def check_event_sourcing(self, images):
            possible_event_sourcing = False
            for image in images:
                if("kafka" in image or "rabbitmq" in image ):
                    possible_event_sourcing = True
            return possible_event_sourcing
    
