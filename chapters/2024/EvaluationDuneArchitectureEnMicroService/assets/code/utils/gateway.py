from dockerfile_parse import DockerfileParser
import yaml

class gateway_analyzer():
    def __init__(self):
        print("Info ", flush=True)

    def detect_gateway(self,dockercompose,directories):
        dockercompose = yaml.safe_load(dockercompose)
        if dockercompose is None:
            return None

        for directory in directories:
            if directory.lower()=="gateway"   or "gateway" in directory.lower():              
                return True
            
        gateway_detected = False
        
        for service_name, service_config in dockercompose.get('services', {}).items():
            if service_name.lower()=="gateway" or "gateway" in service_name.lower():
                gateway_detected = True
                break

        return gateway_detected