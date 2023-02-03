from os import PathLike

import yaml


def yaml_parse(filename: str | PathLike[str]) -> {}:
    """Parse a yaml file"""
    with open(filename, "r") as cin:
        return yaml.safe_load(cin)
