from dockerfile_parse import DockerfileParser
import yaml

class cicd_analyzer():

    def __init__(self):
        print("Info ", flush=True)

    def has_Jenkinsfile(self, repository):
        cicd_files=[]
        contents = repository.get_contents("")
        for content in contents:
            print(content.name, flush=True)
            if  ('jenkinsfile' in content.name.lower())  or ('travis' in content.name.lower()) or ('circle' in content.name.lower()) or ('pipeline' in content.name.lower()) or ('ci' in content.name.lower()) or ('cd' in content.name.lower()) :

                cicd_files.append(content)

            
        return cicd_files

    def check_services_in_CI(self, repository,directories):
        services = []
        cpt=0
        # Obtenez le contenu du fichier 'docker-compose.yml'
        file_contents = self.has_Jenkinsfile(repository)
        ##print("file_content : " + str(file_content), flush=True)

        print(len(file_contents), flush=True)
        if len(file_contents)==0:
                return "No CI/CD present"


        for file_content in file_contents:

            if file_content.content is not None:
                file = file_content.decoded_content.decode("utf-8")
            else:
                file = None

            ## parse file_content to a string
            if file is not None :
                for directory in directories:
                    for line in file.lower():
                        if directory.lower() in file.lower() or line.lower() in directory.lower():
                            services.append(directory)
                            cpt+=1


        print(cpt)
        if cpt==0:
            if len(file_contents)>0:
                return "CI/CD present but no microservices found in it"
            return None
                
        else:
            return "microservices touched by CI/CD"



