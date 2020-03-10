from pydriller import RepositoryMining

commitMessage = []
commitDate = []
commitModification = []
repoSplit = []
fileSplit = []
evolComplexity = []

repoPath = '../../../M1/Takenoko'



def recupFromRepo(repoPath):

    file = open("out.txt", "w")
    for commit in RepositoryMining(repoPath).traverse_commits():
        #file.write('Message {} , date {} , includes {} modified files'.format(commit.msg, commit.committer_date, len(commit.modifications)))
        date = str(commit.committer_date)
        date = date.split(" ")[0]
        #commitDate.append(date)

        nbModification = len(commit.modifications)
        commitModification.append(nbModification)
        #file.write('date {} , ModifiedFiles {}\n'.format(commit.committer_date, len(commit.modifications)))
        file.write('date {} ModifiedFiles {}\n'.format(date, nbModification))

    file.close()


def decoupeResult(file):
    with open(file) as fp:
       line = fp.readline()
       cnt = 1
       while line:
           print(line)
           repoSplit.append(line.split(" , "))
           line = fp.readline()
           cnt += 1


def getNombreCommitTotal(repoPath):
    res = 0
    for commit in RepositoryMining(repoPath).traverse_commits():
        res += 1
    print(res)


def getEvolComplexiteCyclomatic(repoPath):
    #file = open("complexity.txt", "w")
    for commit in RepositoryMining(repoPath).traverse_commits():

        date = str(commit.committer_date)
        date = date.split(" ")[0]

        for modified in commit.modifications:
            complexity = modified.complexity
            fileName = modified.filename

            if(getExtension(fileName)):
                evolComplexity.append("{} {} {}\n".format(fileName, complexity, date))
            #evolComplexity.append("{} complexity {} date {}\n".format(fileName, complexity, date))
            #file.write("{} complexity {} date {}\n".format(fileName, complexity, date))

    evolComplexity.sort()
    file = open("complexityTest.txt", "w")
    for element in evolComplexity:
        file.write(element)
    file.close()


def getExtension(filename):
    splitFilename = filename.split(".")
    if splitFilename[1] == "py" :
        return True
    else :
        return False





def main():
    #recupFromRepo(repoPath)
    #decoupeResult("out.txt")
    #getNombreCommitTotal(repoPath)
    getEvolComplexiteCyclomatic(repoPath)


main()
