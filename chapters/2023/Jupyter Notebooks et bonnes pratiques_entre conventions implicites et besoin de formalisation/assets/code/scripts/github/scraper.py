import io
import mmap
import os
import shutil
import time
import zipfile
from datetime import datetime
from typing import Dict

import requests
import tomli_w
from requests import Response

# Constants
TMP_DIRECTORY_SUFFIX = "-tmp"
OWNER = "owner"
NAME = "name"

# https://github.com/search?l=&o=desc&q=stars%3A%22%3E+1000%22+language%3A%22Jupyter+Notebook%22&s=stars&type=Repositories

repositories = []


def create_toml(directory: str, repository: Dict[str, str]) -> None:
    for root, dirs, files in os.walk(directory):
        for current_file in files:
            if current_file.lower().endswith('.ipynb') and not os.path.isfile(f"{directory}/{current_file}.toml"):
                with open(f"{directory}/{current_file[:-6]}.toml", "wb") as file:
                    file_creation = os.path.getmtime(f"{directory}/{current_file}")
                    tomli_w.dump({'title': os.path.basename(file.name),
                                  'metadata':
                                      {
                                          'path': f"{os.path.relpath(file.name)}",
                                          'source': f"https://github.com/{repository[OWNER]}/{repository[NAME]}",
                                          'author': repository[OWNER],
                                          'date': datetime.fromtimestamp(file_creation).strftime("%d/%m/%Y")
                                      }
                                  }, file)


def file_contains_code(location_from) -> bool:
    if os.stat(location_from).st_size == 0:
        return False
    with open(location_from, 'rb', 0) as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as s:
        if s.find(b'"cell_type": "code"') != -1:
            # Code found, the Jupyter Notebook contains code.
            return True
        # Code not found, the Jupyter Notebook does not contain code.
        return False


def move_ipynb(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    for root, dirs, files in os.walk(directory + TMP_DIRECTORY_SUFFIX):
        for current_file in files:
            if current_file.lower().endswith('.ipynb'):
                location_from = f"{root}/{current_file}"
                location_to = f"{directory}/{current_file}"
                if not os.path.isfile(location_to) and file_contains_code(location_from):
                    stat = os.stat(location_from)
                    os.rename(location_from, location_to)
                    os.utime(location_to, (stat.st_atime, stat.st_mtime))


def delete_directory(directory: str) -> None:
    shutil.rmtree(directory)
    print(f"{directory} deleted.")


def clean_repository(directory: str) -> None:
    # Move all the .ipynb files to the final directory.
    move_ipynb(directory)

    # Delete the temporary directory.
    delete_directory(directory + TMP_DIRECTORY_SUFFIX)


def download_zip(repository: Dict[str, str]) -> Response:
    headers = {"Accept": "application/vnd.github+json"}
    if os.environ.get('GITHUB_TOKEN') is not None:
        headers['Authorization'] = f"Bearer {os.environ.get('GITHUB_TOKEN')}"
    response = requests.get(f"https://api.github.com/repos/{repository[OWNER]}/{repository[NAME]}", headers)
    if response.status_code != 200:
        raise Exception("Error while getting the most popular repositories on GitHub.")

    json_response = response.json()
    # print(f"json_response = {json_response}")
    default_branch = json_response['default_branch']
    download_url = f"https://github.com/{repository[OWNER]}/{repository[NAME]}/archive/refs/heads/{default_branch}.zip"
    print(f"Downloading {download_url}")
    return requests.get(download_url)


def parse_links(config_file) -> None:
    with open(config_file) as f:
        for link in f.readlines():
            link = link.strip().split('/')
            owner = link[3]
            name = ''.join(link[4:])
            repositories.append({OWNER: owner, NAME: name})


def get_directory_str(repository: Dict[str, str]) -> str:
    if len(repository) == 2:
        return f"{repository[OWNER]}-{repository[NAME]}"
    return "unknown-repository"


def extract_zip(repository: Dict[str, str], output_dir: str) -> None:
    directory = get_directory_str(repository)
    zip_response = download_zip(repository)
    print("Extracting archive...")

    zip_file = zipfile.ZipFile(io.BytesIO(zip_response.content))
    for zi in zip_file.infolist():
        zip_file.extract(zi, path=f"{output_dir}/{directory}{TMP_DIRECTORY_SUFFIX}")
        date_time = time.mktime(zi.date_time + (0, 0, -1))
        try:
            os.utime(f"{output_dir}/{directory}{TMP_DIRECTORY_SUFFIX}/{zi.filename}", (date_time, date_time))
        except Exception as e:
            # Handle weird filename.
            print(e)
    zip_file.close()

    print(f"{directory} directory extracted.")


def get_repositories(config_file: str, output_dir: str = "notebooks/github") -> None:
    """
    Retrieve the notebooks of the top 10 most stars repositories on GitHub.
    """
    parse_links(config_file)

    for repository in repositories:
        directory = get_directory_str(repository)
        print(directory)

        if os.path.exists(f"{output_dir}/{directory}"):
            print(f"{repository[OWNER]} already created, this GitHub repository is skipped.")
            continue
        if os.path.exists(f"{output_dir}/{directory}{TMP_DIRECTORY_SUFFIX}"):
            print(f"{directory}{TMP_DIRECTORY_SUFFIX} already created, skip the download "
                  f"part and retrieves the current notebooks.")
        else:
            extract_zip(repository, output_dir)

        clean_repository(f"{output_dir}/{directory}")
        create_toml(f"{output_dir}/{directory}", repository)


if __name__ == '__main__':
    get_repositories('scripts/github/notebooks_top.txt')
