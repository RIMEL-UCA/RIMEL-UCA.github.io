from graphviz import Digraph
from abc import ABC, abstractmethod
import shutil


ERROR_EDGE = '?'


class GraphBuilder(ABC):
    def __init__(self, data):
        self._data = data

    @abstractmethod
    def generate(self, filename, **kwargs):
        pass


class DotGraphBuilder(GraphBuilder):
    """
    Doc for styling: https://graphviz.org/doc/info/attrs.html
    Check this for example: https://www.graphviz.org/gallery/
    """
    __style = {
        'seq': {'style': 'dotted'},
        # Inter-job relations
        'up/down': {'decorate': 'false'},
        'release': {'style': 'dashed'},

        # Intra-job relations
        'file': {'decorate': 'false'},
        'lang': {'style': 'dashed'}
    }

    def __init__(self, data: dict[str, list[tuple]]):
        """
        :param data: Expected format: `{'node1': [('dest_node', 'link_type', 'msg')], 'node2': []}`
        """
        super().__init__(data)
    
    @staticmethod
    def clean_node_name(name):
        return name.replace('${{ ', '{').replace(' }}', '}').replace('env.', '').replace(":", '')

    def generate(self, filename, file_format="png", output_dir='images', **kwargs):
        graph = Digraph(filename=f"{filename}.dot", format=file_format, **kwargs)

        for node in self._data.keys():
            if 'Anonymous' in node:
                graph.node(self.clean_node_name(node), color='orange', shape='rect')
            else:
                graph.node(self.clean_node_name(node), shape='rect')

        if ERROR_EDGE in self._data.keys():
            graph.node(ERROR_EDGE, shape='ellipse', color='red', fontcolor='red')

        for node, edges in self._data.items():
            for dest, edge_type, msg in edges:
                if edge_type not in self.__style.keys():
                    raise KeyError("Edge type unknown %s" % edge_type)

                style = {}
                if dest == ERROR_EDGE:
                    style['color'] = 'orange'
                elif node == ERROR_EDGE:
                    style['color'] = 'orange'

                if "Missing" in msg or 'Unknown' in msg:
                    style['color'] = 'red'
                    style['fontcolor'] = 'red'

                graph.edge(self.clean_node_name(node), self.clean_node_name(dest), xlabel=msg, **self.__style[edge_type], **style)

        resfile = graph.render(directory=output_dir)

        shutil.move(resfile, resfile.replace(".dot", ''))
