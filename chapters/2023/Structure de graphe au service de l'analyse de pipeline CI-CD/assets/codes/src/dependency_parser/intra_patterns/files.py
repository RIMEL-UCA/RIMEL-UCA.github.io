from src.dependency_parser.strategies import IntraScanStrategy
import os
from logging import getLogger

def file4dockercompose(cmd, default = 'docker-compose.yml', options = ['-f', '--file']):
    res = []
    cmd = cmd.split(' ')

    for i in range(len(cmd)):
        if cmd[i] in options: # Then the following arg is a custom file usage
            res += [cmd[i + 1]]

    return res if res else [default]


class Strategy(IntraScanStrategy):
    __cmds = {
        'npm i': 'package.json', # Will be detected in the case of a `npm install`
        'npm run': 'package.json', # Will be detected in the case of a `npm install`
        'npm ci': ['package.json', 'package-lock.json'],
        'npm clean-install': ['package.json', 'package-lock.json'],
        'mvn': 'pom.xml',
        'docker-compose': file4dockercompose
    }

    def __init__(self, project_path):
        super().__init__(project_path)
        self._logger = getLogger(Strategy.__name__)

    def parse(self, job: dict, graph: dict):
        checkout = False

        for step in job['steps']:
            if 'name' not in step:
                step['name'] = step['id'] if 'id' in step else (step['uses'] if 'uses' in step else 'Anonymous step')
            
            if step['name'] not in graph:
                graph[step['name']] = []

            # Does the step make a checkout
            if 'uses' in step and 'actions/checkout' in step['uses']:
                checkout = step['name']
            
            if 'run' in step:
                self.__check_file_usage(step['name'], step['run'], checkout, graph)
    
    def __check_file_exists(self, _file):
        return os.path.exists(f"{self._path}/{_file}")
    
    @staticmethod
    def _dep_maker(step, checkout, dep, exists: bool, graph: dict):
        if not checkout:
            checkout = '?'
        
        deps = []
        if checkout in graph:
            deps = graph[checkout]
        else:
            graph[checkout] = deps
        
        link = (step, 'file', dep if exists else f"Missing file {dep}")
        if link not in deps:
            deps += [link]

    def __check_file_usage(self, name, run, checkout, graph: dict):
        for cmd, _file in self.__cmds.items():
            if cmd in run:
                if type(_file) is str:
                    self._dep_maker(name, checkout, _file, self.__check_file_exists(_file), graph)
                elif type(_file) is list:
                    for f in _file:
                        self._dep_maker(name, checkout, f, self.__check_file_exists(f), graph)
                else:
                    files = _file(run)
                    for f in files:
                        self._dep_maker(name, checkout, f, self.__check_file_exists(f), graph)
        
        run = run.replace("\n", " ")
        for arg in run.split(' '):
            if os.path.isfile(f"{self._path}/{arg}"):
                self._dep_maker(name, checkout, arg, self.__check_file_exists(arg), graph)
