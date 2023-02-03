from src.dependency_parser.strategies import IntraScanStrategy
import os


# Those methods / keywords comes with Ubuntu and will not be treated yet
excluded_methods = [
    'cd', 'cp', 'mv'
    'if', 'else', 'fi', 
    'ls',
    'echo',
    'chmod', 
    'for', 'do', 'done', 
    'case', 'esac', 
    'tar',
    'curl',
    'sed',
    'docker-compose', 'docker',
    'export',
    '{', '}',
    "printf", "local", "[[",
    "function",
    "cmake"
]

setup_effects = {
    'actions/setup-node': [
        'npm', 'node'
    ],
    'actions/setup-python': [
        'python'
    ],
    'actions/setup-java': [
        'java'
    ]
}


class Strategy(IntraScanStrategy):
    def __init__(self, project_path):
        super().__init__(project_path)
    
    def parse(self, job: dict, graph: dict):
        for step in job['steps']:
            if 'name' not in step:
                step['name'] = step['id'] if 'id' in step else (step['uses'] if 'uses' in step else 'Anonymous step')

            if step['name'] not in graph:
                graph[step['name']] = []
        
        setups = self.__get_setup_actions(job['steps'])

        for step in job['steps']:
            if 'run' in step:
                self.__check_cmd_exists(step['name'], step['run'], setups, graph)
    
    @staticmethod
    def __get_setup_actions(steps: list):
        return {
            step['uses'][:step['uses'].index("@")]: step['name']
            for step in steps
            if 'uses' in step and 'actions/setup-' in step['uses']
        }
    
    def __check_cmd_exists(self, name, run, setups, graph: dict):
        for op, substitue in {'&&': '\n', '||': '\n'}.items():
            run = run.replace(op, substitue) # To avoid splitting N times

        methods = []
        function_customs = []
        for call in run.split('\n'):
            call = [arg for arg in call.split(" ") if arg]

            if call and call[0] == 'function':
                function_customs += [call[1].replace("()", '')]

            # Get each method called only once
            if call and call[0] not in methods:
                methods += [call[0]]

        methods = [method for method in methods if method not in excluded_methods and '=' not in method]
        methods = [method for method in methods if method not in function_customs]
        methods = [method for method in methods if not os.path.isfile(f"{self._path}/{method}")]
        
        if not methods:
            return
        
        for method in methods:
            setter = next(filter(lambda k: method in setup_effects[k], setup_effects.keys()), None)
            src, msg = '', ''
            if setter is None:
                src, msg = '?', f'Unknown method \"{method}\"'
            elif setter in setups:
                src, msg = setups[setter], method
            else:
                src, msg = '?', f"No setup for \"{method}\""
            
            deps = []
            if src in graph:
                deps = graph[src]
            else:
                graph[src] = deps
            
            deps += [(name, 'lang', msg)]
