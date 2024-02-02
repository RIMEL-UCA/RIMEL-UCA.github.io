from dockerfile_parse import DockerfileParser
import yaml

class masterslave_analyzer():
    def __init__(self):
        print("Info ", flush=True)

    
        

    def detect_master_slave_replication(self, repository,dockercompose):
        file_content = dockercompose
        if file_content is None:
            return None

        replication_keywords = ['master', 'slave', 'replica']

        for line in file_content.split('\n'):
            for keyword in replication_keywords:
                if keyword in line.lower():
                    # Vous pouvez ajuster cette logique selon vos besoins
                    return True  # Détecté une configuration de réplication maître-esclave

        return False  # Aucune configuration de réplication détectée