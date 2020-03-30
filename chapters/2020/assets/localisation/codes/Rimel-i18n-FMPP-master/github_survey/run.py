import shutil
from multiprocessing.pool import Pool

from tqdm import tqdm
from git.repo.base import Repo

from translation_detection import check_if_localised

def checkGit(git):
    print(git)
    try:
        name = git.split("/")[-1].split(".")[0]
        repo = Repo.clone_from(git, name)
        check =  check_if_localised(repo.working_dir)
        shutil.rmtree(repo.working_dir)
    except :
        return False
    return check

with open("./top_gits.txt") as gits:
    localized = list(filter(checkGit, tqdm(gits)))
    count = len(localized)
    print(count)
    print(localized)
    with open("stared_git", "w") as count:
        count.writelines(localized)
