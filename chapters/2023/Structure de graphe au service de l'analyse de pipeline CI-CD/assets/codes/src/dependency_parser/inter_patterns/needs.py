from src.dependency_parser.utils import job_names
from src.dependency_parser.strategies import InterScanStrategy


class Strategy(InterScanStrategy):
    @staticmethod
    def _get_dependencies(job):
        return job['needs'] if 'needs' in job else []
    
    @staticmethod
    def _get_node(node, graph):
        if node not in graph:
            graph[node] = []
        
        return graph[node]

    def parse(self, jobs: dict, graph: dict):
        for name, job in jobs.items():
            deps = []
            if name in graph:
                deps = graph[name]
            else:
                graph[name] = deps
            
            needs = self._get_dependencies(job)
            if type(needs) is str:
                deps = self._get_node(needs, graph)
                deps += [(name, 'seq', '')]
                graph[needs] = deps
                # deps += [(needs, 'seq', '')]
            else:
                for need in needs:
                    deps = self._get_node(need, graph)
                    deps += [(name, 'seq', '')]
                    graph[need] = deps
                # deps += [(need, 'seq', '') for need in needs]


def detect(yml):
    return { job: [] for job in job_names(yml) }

def links_dependencies(yml, job_map):
    def __needs (yaml, job):
        return yaml['jobs'][job]['needs'] if 'needs' in yaml['jobs'][job] else []
    
    return {
        job: [(need, 'seq', '') for need in __needs(yml, job)] 
        for job in job_map.keys()
    }

def build_graph(dot, dep):
    for node in dep.keys():
        dot.node(node)
    
    for job, dependencies in dep.items():
        for dep, _type, _ in dependencies:
            if _type == 'seq':
                dot.edge(dep, job)
