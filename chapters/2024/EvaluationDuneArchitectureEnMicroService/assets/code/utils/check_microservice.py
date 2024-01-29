from dockerfile_parse import DockerfileParser
import yaml

class microservice_keywords():

    def __init__(self):
        print("Info ", flush=True)

    def identify_microservice_keywords(self, repository):
        keywords = ["microservice","service-oriented","service oriented","service_oriented"]
        contents = repository.get_contents("")

        ## check first in readme
        for content in contents:
            if content.name.lower() == "readme.md":
                readme = content.decoded_content.decode("utf-8")
                for keyword in keywords:
                    if keyword in readme.lower():
                        return True
                    

        ## check in name of the repo
        repo_name = repository.name
        for keyword in keywords:
            if keyword in repo_name.lower():
                return True
            
        ## check in description
            
        repo_description = repository.description
        for keyword in keywords:
            if keyword in repo_description.lower():
                return True
            
        ## check in topics
            
        repo_topics = repository.get_topics()
        for keyword in keywords:
            if keyword in repo_topics:
                return True
            
        ## check in files
        
        for content in contents:
            if "dbs" in content.name.lower():
                dbs = content.decoded_content.decode("utf-8")
                for keyword in keywords:
                    if keyword in dbs.lower():
                        return True

            
        return False





