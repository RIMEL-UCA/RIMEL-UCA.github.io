import json
from pathlib import Path


def read_files(directory: str) -> list:
    """
    Read result files generated after the analysis.
    :param directory: the path of the result files.
    :return: a list of json read.
    """
    results = []
    for n in Path(directory).glob('*.json'):
        results.append(json.loads(n.read_text(encoding="UTF-8")))
    return results
