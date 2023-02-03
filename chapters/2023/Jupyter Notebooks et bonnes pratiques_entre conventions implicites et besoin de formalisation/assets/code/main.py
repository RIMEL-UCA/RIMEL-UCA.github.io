from pathlib import Path
from typing import Optional

import os
import sys
import linecache
import tracemalloc

import typer

from scripts import analysis
from scripts.diagrams import (
    analysis_commons_error,
    diagram_cells_github_vs_kaggle,
    diagram_cells_references_vs_lower,
    diagram_cells_popular_vs_lower,
    diagram_pylint_comparison,
    diagram_pylint_trend,
    diagram_pylint_zero_github_vs_kaggle,
)
from scripts.github import scraper as scraper_github
from scripts.kaggle import scraper as kaggle_scraper

app = typer.Typer()

def __analyse_file(filepath: Path, output_dir: str = 'results'):
    print(f"Running for {filepath}...", end="")
    result = analysis.run_analysis(notebook_name=filepath.with_name(filepath.stem), output_dir=output_dir, verbose=False)
    print("done" if result else "error")
    return result


def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


@app.command()
def analyze(directory: Optional[str] = typer.Argument('notebooks/'),
            output_dir: Optional[str] = typer.Argument('results/')):
    print(f"Running analysis in directory: {directory}")
    from multiprocessing import Pool
    sys.setrecursionlimit(10000) 
    # tracemalloc.start()
    for filepath in Path(directory).glob('**/*.ipynb'):
        print(f"Running for {filepath}...", end="")
        result = analysis.run_analysis(notebook_name=filepath.with_name(filepath.stem), output_dir=output_dir, verbose=False)
        print("done" if result else "error")
    with Pool(5) as p:
        p.map(__analyse_file, list(Path(directory).glob('**/*.ipynb')))
    # snapshot = tracemalloc.take_snapshot()
    # display_top(snapshot, limit=10)


@app.command()
def retrieve_common_errors(directory):
    print(analysis_commons_error.get_error_stats(directory))


@app.command()
def plot_cells_github_vs_kaggle(directory: str, title: str = typer.Argument('Number of markdown per code cells')):
    diagram_cells_github_vs_kaggle.plot_diagram(directory, title)


@app.command()
def plot_cells_references_vs_lower():
    diagram_cells_references_vs_lower.plot_diagram()


@app.command()
def plot_cells_popular_vs_lower():
    diagram_cells_popular_vs_lower.plot_diagram()


@app.command()
def plot_pylint_comparison(directory_a: str, directory_b, title: str = typer.Argument('Score of notebooks')):
    diagram_pylint_comparison.plot_diagram(directory_a, directory_b, title)


@app.command()
def plot_pylint_trend(directory: str, title: str = typer.Argument('Score of notebooks')):
    diagram_pylint_trend.plot_diagram(directory, title)


@app.command()
def plot_pylint_zero_github_vs_kaggle(directory: str, title: str = typer.Argument('Number of pylint score=0')):
    diagram_pylint_zero_github_vs_kaggle.plot_diagram(directory, title)


@app.command()
def scrap_github(config_filepath: str = typer.Argument('scripts/github/notebooks.txt'),
                 output_dir: Optional[str] = typer.Argument('notebooks/github')):
    print(f"Scrapping github using config: {config_filepath}")
    scraper_github.get_repositories(config_filepath, output_dir)


@app.command()
def scrap_kaggle(config_filepath: str = typer.Argument('scripts/kaggle/notebooks.txt')):
    print(f"Scrapping kaggle using config: {config_filepath}")
    kaggle_scraper.list_top_notebook(config_filepath)
    kaggle_scraper.scrap(config_filepath)


if __name__ == "__main__":
    app()
