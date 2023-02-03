from src.parser import yaml_parse
from importlib import import_module
from logging import getLogger
import graphviz
import logging
import os


__parsers = {}
_PATH = os.getcwd() + '/src/dependency_parser'


def _find_parsers():
    global __parsers
    for dir_path in list(filter(os.path.isdir, ['%s/%s' % (_PATH, item) for item in os.listdir(_PATH)])):
        directory = dir_path.split("/")[-1]
        if directory == '__pycache__':
            continue
        __parsers[directory] = [
            pyfile.replace('.py', '')
            for pyfile in list(filter(
                lambda f: f.endswith('.py')
            , os.listdir(dir_path)))
            if not pyfile.endswith('__init__.py')
        ]
    getLogger(__name__).debug('parsers: %s' % __parsers)



def generate_graph(file_name, graph):
    """Generate a graphviz image representation of a graph"""
    logger = logging.getLogger(__name__)
    final_filename = file_name.split('/')[-1].replace('.yml', '.dot')
    dot = graphviz.Digraph(filename=final_filename, format="png")
    logger.info(f"Creating {final_filename} images from the graph which come from {file_name}")
    # for each studied patterns, build a graph.
    for key in graph.keys():
        for pattern in graph[key]:
            graph[key][pattern]['graph_loader'](dot,graph[key][pattern]['dep'])

    #render graph
    logger.info('Resulting file: %s' % dot.render(directory='images'))


def make_graph(file_name):
    """This function first detects patterns of a file, then processed them and
    make a graph that belongs to inter and intra dependencies relation"""
    if __parsers == {}:
        _find_parsers()

    yml = yaml_parse(file_name)
    graph = {}
    # dynamic import of a file, it could belong to inter package or intra
    def import_pattern(type_dep, module):
        return import_module(f"src.dependency_parser.{type_dep}.{module}")

    # load all required modules and assign to graph type : intra || inter -> a dict with 2 keys module and dependencies
    for key, modules in __parsers.items():
        getLogger(__name__).debug("Type of dependencies: " + key + ", stategy: " + str(modules))
        modules = {module: import_pattern(key, module) for module in modules}
        # identifies all dependency that match with the current one evaluated
        detected_pattern = {module: modules[module].detect(yml) for module in modules.keys()}

        getLogger(__name__).debug(detected_pattern)

        graph[key] = {module: {'dep': modules[module].links_dependencies(yml, detected_pattern[module]),
                               'graph_loader': modules[module].build_graph} for module in modules.keys()}

    generate_graph(file_name,graph)
