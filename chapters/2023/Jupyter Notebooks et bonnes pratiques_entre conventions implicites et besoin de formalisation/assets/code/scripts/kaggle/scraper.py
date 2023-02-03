import os
import time

os.environ['KAGGLE_CONFIG_DIR'] = f"scripts/kaggle"

import tomli_w
from kaggle.api.kaggle_api_extended import KaggleApi



def list_top_notebook(config_file: str = 'scripts/kaggle/notebooks.txt'):
    api = KaggleApi()
    api.authenticate()
    with open(config_file, 'w') as f:
        f.write('')
    for page in range(1, 4):
        notebooks = api.kernels_list(sort_by="voteCount", language='python', page_size=100, page=page)
        with open(config_file, 'a') as f:
            for notebook in notebooks:
                f.write(str(getattr(notebook, "ref")) + " " + str(getattr(notebook, "lastRunTime")) + "\n")


def scrap(config_file: str = 'scripts/kaggle/notebooks.txt'):
    api = KaggleApi()
    api.authenticate()

    file1 = open(config_file, 'r')
    lines = file1.readlines()
    baseUrl = "https://www.kaggle.com/code/"
    for line in lines:
        element = line.split()[0]
        path = "notebooks/kaggle/" + element.strip().replace('/', '-') + "/"
        author = element.strip().split('/', 1)[0]
        fileName = element.strip().split('/', 1)[1]
        print(f"Saving {element.strip()}...", end="")
        api.kernels_pull(element.strip(), path=path)
        time.sleep(1)
        print("done")
        with open(f"{path + fileName}.toml", "wb") as f:
            print(f"Saving {path + fileName}.toml...", end="")
            tomli_w.dump({'title': fileName,
                          'metadata': {'path': fileName + '.ipynb', 'source': baseUrl + element.strip(), 'author': author,
                                       'date': line.split()[1]}}, f)
            print("done")


if __name__ == '__main__':
    list_top_notebook()
    print(os.environ.get('KAGGLE_CONFIG_DIR'))
    scrap()
