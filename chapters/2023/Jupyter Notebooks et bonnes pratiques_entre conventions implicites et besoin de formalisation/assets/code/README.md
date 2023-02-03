# JupyterNotebookQA

This project uses a [Makefile](./Makefile) for all steps.

## Setup

### Requirements

* Python 3.10 + pip
* Kaggle credentials in code/scripts/kaggle/kaggle.json

To run the analysis, you have to install the [pipenv](https://pipenv.pypa.io/en/latest/index.html) tool as well as the virtual environments using the following Makefile recipes:

### Install pipenv

```bash
$ make install_pipenv
```

### Install with standard dependencies

```bash
$ make install_local_env
```

### Install with standard + popular dependencies

```bash
$ make install_local_popular_env
```

## Run scrappers

```bash
$ make scrap_github
$ make scrap_kaggle
```

## Run analysis

```bash
$ make analyze
```

## Plotting results

To plot the result, you can use the CLI.
You can get the list of available commands using the following command:

```bash
$ make cli_help
```

Plotting commands start with `plot-`. You can retrieve the common errors using the `retrieve-common-errors` CLI command.
