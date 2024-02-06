from dockerfile_parse import DockerfileParser
import yaml

class loadbalancer_analyzer():

    def detect_load_balancer(self,repository, images):
        possible_load_balancing = False

        # Check if images contain 'haproxy' or 'nginx'
        if images and any(("haproxy" in image or "nginx" in image or "traefik" in image) for image in images):
            possible_load_balancing = True

        # Search for nginx.conf
        contents = repository.get_contents("")
        nginx = self.search_for_loadBalancing_conf(repository=repository,contents=contents)

        if nginx is None:
            return possible_load_balancing, "nginx.conf not found"

        # Check for specific keywords in nginx.conf
        keywords_present = self.detect_scalability(nginx)
        if keywords_present:
            return True, "scalability present"
        else:
            return possible_load_balancing, "scalability not present"



    def detect_scalability(self,nginx_content):
            keywords = ["upstream", "server", "listen", "proxy_pass", "proxy_set_header", "proxy_http_version", "location","balance","roundrobin","leastconn","ip_hash","health_check","reload-config","listen","loadBalancer"]

            for keyword in keywords:
                if keyword in nginx_content:
                    return True
                
            return False
                
            
    def search_for_loadBalancing_conf(self,repository,contents, current_path=""):
        for content in contents:
            if content.type == "dir":
                sub_contents = repository.get_contents(content.path)
                path = f"{current_path}/{content.name}" if current_path else content.name
                nginx_conf = self.search_for_loadBalancing_conf(repository,sub_contents, path)
                if nginx_conf:
                    return nginx_conf
            elif content.name.lower() == 'nginx.conf' or 'nginx' in content.name.lower():
                nginx_content = repository.get_contents(content.path).decoded_content.decode("utf-8")
                return nginx_content
            ## haproxy
            elif content.name.lower() == 'haproxy.cfg' or 'haproxy' in content.name.lower():
                nginx_content = repository.get_contents(content.path).decoded_content.decode("utf-8")
                return nginx_content
            ## traefik
            elif content.name.lower() == 'traefik.yml' or 'traefik' in content.name.lower():
                nginx_content = repository.get_contents(content.path).decoded_content.decode("utf-8")
                return nginx_content

        return None

    def detect_scalability_with_keywords(nginx_content):
        keywords = ["upstream", "server", "listen", "proxy_pass", "proxy_set_header", "proxy_http_version", "location","balance","roundrobin","leastconn","ip_hash","health_check","reload-config","listen","loadBalancer"]
        for keyword in keywords:
            if keyword in nginx_content:
                return True
        return False
    
    def process_load_balancer_result(self,result):
        load_balancing_status, message = result
        if load_balancing_status:
            if message == "scalability present":
                return "LoadBalancing and Scalability"
            else:
                return"LoadBalancing and no Scalability"
        else:
            return "Not present"


