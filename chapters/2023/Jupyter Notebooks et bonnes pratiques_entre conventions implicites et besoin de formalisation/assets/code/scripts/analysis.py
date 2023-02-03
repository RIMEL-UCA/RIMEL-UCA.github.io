# import ast
import contextlib
import io
import json
import sys
import tomli
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Tuple
from uuid import uuid1

import flake8
from mypy import api
from pylint.lint import Run, pylinter


def filter_cells_by(data: Dict[str, Any], filter_key: str, filter_value: str) -> Tuple[Dict[str, Any]]:
    return tuple(n for n in data["cells"] if n[filter_key] == filter_value)


def get_cells(data: Dict[str, Any]) -> Tuple[Dict[str, Any]]:
    return tuple(n for n in data["cells"])


def get_code_cells(data: Dict[str, Any]) -> Tuple[Dict[str, Any]]:
    return filter_cells_by(data, "cell_type", "code")


def get_markdown_cells(data: Dict[str, Any]) -> Tuple[Dict[str, Any]]:
    return filter_cells_by(data, "cell_type", "markdown")


def execute_for_each_code_cell(code_cells, func) -> Any:
    Path("tmp/").mkdir(parents=True, exist_ok=True)
    results = []
    for code_cell in code_cells:
        tmp_filename = f"tmp/code_cell_{uuid1()}.py.tmp"
        tmp_path = Path(tmp_filename)
        with open(tmp_filename, mode="w", encoding='utf-8') as f:
            f.writelines(code_cell["source"])
            results.append(func(tmp_filename))
        tmp_path.unlink()
    return results


def run_code_analysis(code_cells):
    Path("tmp/").mkdir(parents=True, exist_ok=True)
    tmp_filename = f"tmp/code_{uuid1()}.tmp.py"
    tmp_path = Path(tmp_filename)
    tmp_path.touch(exist_ok=True)

    total_nb_lines = 0
    for code_cell in code_cells:
        with open(tmp_filename, mode="a", encoding='utf-8') as f:
            source = code_cell["source"] if isinstance(code_cell["source"], str) else '\n'.join(code_cell["source"])
            lines = []
            for s in source.split('\n'):
                if s.strip().startswith('%'):
                    s = s.replace('%', 'pass # %')
                if s.strip().startswith('!'):
                    s = s.replace('!', '# !')
                if s.strip().startswith('pip install '):
                    s = s.replace('pip install ', '# pip install ')
                lines.append(s)
            lines.append('\n')
            total_nb_lines += len(lines) - 1
            f.write('\n'.join(lines))
    with contextlib.redirect_stdout(io.StringIO()):
        pylint_run = Run([tmp_filename], exit=False)
        pylint_result = {
            'score': pylint_run.linter.stats.global_note,
            'count_by_severity': list(pylint_run.linter.stats.by_module.values())[0],
            'count_by_messages': pylint_run.linter.stats.by_msg
        }
        pylinter.MANAGER.clear_cache()
        pylint_run.linter.close()
        # mypy_run = api.run([tmp_filename])
    # mypy_output_lines = [line for line in mypy_run[0].strip().split('\n') if ': note: ' not in line and 'module is installed, but missing library stubs or py.typed marker' not in line][:-1]
    # mypy_nb_errors = len(tuple(line for line in mypy_output_lines if ': error: ' in line))
    mypy_result = {
        'score': -1, # 1 - int(mypy_nb_errors) / total_nb_lines if mypy_nb_errors > 0 else 100,
        'count_by_severity': {}, # dict(Counter(e.split(': ')[1] for e in mypy_output_lines)),
        'count_by_messages': {}, # dict(Counter(e.split('  [')[1][:-1] for e in mypy_output_lines)),
    }
    tmp_path.unlink()

    return pylint_result, mypy_result


def compute_profile(data):
    cells = get_cells(data)
    return [
        {
            'cell_type': cell['cell_type'],
            'nb_lines': sum(len(p.strip().split('\n')) for p in cell['source']),
            # 'imports': 0 if cell['cell_type'] != 'code' else len(n for n in ast.iter_child_nodes(ast.parse('\n'.join('\n'.join(s for s in cell['source'] if not s.startswith('%'))))) if isinstance(n, ast.Import, ast.ImportFrom))
        } for cell in cells
    ]


def run_analysis(
        notebook_name: str = "notebooks/github/notebook",
        output_dir: str = "results",
        verbose: bool = True,
        log_errors: bool = True
) -> bool:
    display = print if verbose else lambda *args, **kwargs: None
    display_error = print if log_errors else lambda *args, **kwargs: None
    if not Path(f"{notebook_name}.ipynb").exists():
        display_error(f"File {notebook_name}.ipynb does not exist. Skipping.")
        return False
    
    if not Path(f"{notebook_name}.toml").exists():
        display_error(f"File {notebook_name}.toml does not exist. Skipping.")
        return False
    with open(f"{notebook_name}.toml", mode="rb") as f:
        notebook_metadata = tomli.load(f)
    
    try:
        with open(f"{notebook_name}.ipynb", encoding="utf-8") as f:
            notebook = json.load(f)
    except json.JSONDecodeError:
        with open(
            f"results/{notebook_metadata['metadata']['author']}_{notebook_metadata['title']}.json",
            mode="w",
            encoding="utf-8",
        ) as f:
            json.dump({'error': 'invalid JSON'}, f, indent=4)
        display("\nerror")
        return False

    display(f"========= Analysing {notebook_metadata['title']} =========\n")

    display("Analysing notebook structure...")
    try:
        code_cells = get_code_cells(notebook)
        markdown_cells = get_markdown_cells(notebook)
    except KeyError:
        with open(
            f"results/{notebook_metadata['metadata']['author']}_{notebook_metadata['title']}.json",
            mode="w",
            encoding="utf-8",
        ) as f:
            json.dump({'error': 'unsupported notebook structure'}, f, indent=4)
        display("\nerror")
        return False

    profile = compute_profile(notebook)

    display("Analysing code quality (pylint, mypy)")
    pylint_score, mypy_score = run_code_analysis(code_cells)

    display("\nExporting results...", end="")

    results = {
        "title": notebook_metadata["title"],
        "notebook": f"{notebook_name}.ipynb",
        "metadata": f"{notebook_name}.toml",
        "metrics": {
            "nb_code_cells": len(code_cells),
            "nb_markdown_cells": len(markdown_cells),
            # TODO: check !pip use for dependencies installation
            "dependencies_installation_detected": "!pip " in code_cells[0]["source"] if code_cells else False,
            "code_quality": {
                "pylint_score": pylint_score,
                "mypy_score": mypy_score,
            },
        },
        "profile": profile,
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(
            f"{output_dir}/{notebook_metadata['metadata']['author']}_{notebook_metadata['title']}.json",
            mode="w",
            encoding="utf-8",
    ) as f:
        json.dump(results, f, indent=4)

    display(json.dumps(results, indent=4))

    display("\ndone")
    return True


if __name__ == '__main__':
    run_analysis(notebook_name="notebooks/github/ageron-handson-ml/01_the_machine_learning_landscape", output_dir="results")
