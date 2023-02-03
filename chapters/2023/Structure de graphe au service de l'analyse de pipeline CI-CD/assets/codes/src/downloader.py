from src.const import TMP_DIR
import os.path
import logging
from git import Repo
from src.progressbar import GitRemoteProgress


class Downloader:
    def __init__(self, repository_path: str | list[str]):
        self.__git_url = repository_path
        self.logger = logging.getLogger(Downloader.__name__)

    def __download_repo(self, repo_url, output_path):
        """Download a git repository to the output directory"""
        # Assert that the url is a git clone url
        if '.git' not in repo_url:
            print("Given path is not a valid git url")
            return
        repo_name = repo_url.split(os.path.sep)[-1][:-len('.git')]
        self.logger.info("Downloading %s to %s" % (repo_url, output_path + repo_name))
        Repo.clone_from(repo_url, output_path + repo_name, progress=GitRemoteProgress())
        self.logger.debug('Downloading of %s done !' % repo_name)

    def download(self, output_path: str = f"{TMP_DIR}/"):
        """Download git repository"""
        if type(self.__git_url) is list:
            for repo_url in self.__git_url:
                self.__download_repo(repo_url, output_path)
        else:
            self.__download_repo(self.__git_url, output_path)
