install_pipenv:
	pip install --user pipenv

install_local_env:
	pipenv install
	pipenv install --dev

install_local_popular_env:
	pipenv install --categories popular

scrap_github:
	pipenv run 'python' 'main.py' 'scrap-github' 'scripts/github/notebooks_top.txt' 'notebooks/github/popular'
	pipenv run 'python' 'main.py' 'scrap-github' 'scripts/github/notebooks_references.txt' 'notebooks/github/references'
	pipenv run 'python' 'main.py' 'scrap-github' 'scripts/github/notebooks_lower.txt' 'notebooks/github/lower'

scrap_kaggle:
	pipenv run 'python' 'main.py' 'scrap-kaggle' 'scripts/kaggle/notebooks_top.txt'

analyze:
	pipenv run 'python' 'main.py' 'analyze' 'notebooks/' 'results/'

cli_help:
	pipenv run 'python' 'main.py' '--help'
