from abc import ABC, abstractmethod


class InterScanStrategy(ABC):
    @abstractmethod
    def parse(self, jobs: dict, graph: dict):
        pass


class IntraScanStrategy(ABC):
    def __init__(self, project_path):
        self._path = project_path

    @abstractmethod
    def parse(self, job: dict, graph: dict):
        pass
