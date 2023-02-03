from src.const import GITHUB_ACTION_PATH
from src.parser import yaml_parse
from logging import getLogger
from importlib import import_module
import os


_path = os.getcwd() + '/src/dependency_parser'


def _get_parsing_strategies(inter: str = f'{_path}/inter_patterns', intra: str = f'{_path}/intra_patterns'):
    is_pyfile = lambda f: f.endswith('.py')

    def get_pyfile_from(directory, excluded = ['__init__.py', '__main__.py']):
        return [pyfile[:-3] for pyfile in list(filter(is_pyfile, os.listdir(directory))) if pyfile not in excluded]

    return get_pyfile_from(inter), get_pyfile_from(intra)



class Scanner:
    def __init__(self, project_name: str, project_path, actions = []):
        self._project = project_name
        self._path = project_path
        self._actions = actions

        self._results = {}

        self._logger = getLogger(Scanner.__name__)

        # Get the parsing strategies
        self._inter, self._intra = _get_parsing_strategies()

        self._logger.info("Project %s" % (self._project))
        self._logger.debug(f"Strategies: {self._inter}, {self._intra}")

        # If no action are given to parse
        if actions == []:
            # Automatic discovery
            self._actions = os.listdir(f"{project_path}/{GITHUB_ACTION_PATH}")
        
        self._logger.info(f"Actions {self._actions}")
    
    @staticmethod
    def _instanciate_strategy(mod, *args, **kwargs):
        return import_module(mod).Strategy(*args, **kwargs)
    
    @staticmethod
    def _instanciate_inter_strategy(name, *args, **kwargs):
        return Scanner._instanciate_strategy(f'src.dependency_parser.inter_patterns.{name}', *args, **kwargs)
    
    @staticmethod
    def _instanciate_intra_strategy(name, *args, **kwargs):
        return Scanner._instanciate_strategy(f'src.dependency_parser.intra_patterns.{name}', *args, **kwargs)
    
    def parse(self):
        """Parse the project to build a dependency graph"""
        for action in self._actions:
            self._results[action] = self._parse_file(f"{GITHUB_ACTION_PATH}/{action}")
    
    def _parse_file(self, action):
        self._logger.debug(f"Parsing {action}")
        yml = yaml_parse(f"{self._path}/{action}")
        result = {
            'inter': {}, # For a new graph
            'intra': {}  # To bind each intra-graph to a name
        }

        for inter_pattern in self._inter:
            self._instanciate_inter_strategy(inter_pattern).parse(yml['jobs'], result['inter'])

        for name, job in yml['jobs'].items():
            result['intra'][name] = self._parse_job(job)
            
        return result
    
    def _parse_job(self, job):
        graph = {}

        for intra_pattern in self._intra:
            self._instanciate_intra_strategy(intra_pattern, self._path).parse(job, graph)
        
        return graph

    def get_results(self, action=None):
        return self._results if action is None else self._results[action]
