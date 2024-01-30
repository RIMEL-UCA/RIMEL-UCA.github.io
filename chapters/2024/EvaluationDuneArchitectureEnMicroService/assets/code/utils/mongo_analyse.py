from dockerfile_parse import DockerfileParser
import yaml

class mongo_analyzer():
    def __init__(self):
        print("Info ", flush=True)

    def detect_mongo_replication(self,dockercompose):
        dockercompose = yaml.safe_load(dockercompose)
        if dockercompose is None:
            return None

        replication_keywords = [' --replSet rs0']
        replication_detected = False

        for service_name, service_config in dockercompose.get('services', {}).items():
            if 'image' in service_config and ('mongo' in service_config['image'])  :
                if 'command' in service_config:
                    command = service_config['command']
                    for keyword in replication_keywords:
                        if keyword in command:
                            replication_detected = True
                            break

        return replication_detected
