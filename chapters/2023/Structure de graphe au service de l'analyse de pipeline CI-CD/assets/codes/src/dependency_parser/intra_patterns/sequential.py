from src.dependency_parser.strategies import IntraScanStrategy


class Strategy(IntraScanStrategy):
    def __init__(self, project_path):
        super().__init__(project_path)
    
    def parse(self, job: dict, graph: dict):
        last = None

        for step in job['steps']:
            if 'name' not in step:
                step['name'] = step['id'] if 'id' in step else (step['uses'] if 'uses' in step else 'Anonymous step')
            
            if step['name'] not in graph:
                graph[step['name']] = []

            if last is not None:
                graph[last] += [(step['name'], 'seq', '')]

            last = step['name']
