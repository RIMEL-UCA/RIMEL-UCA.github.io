from pydriller import RepositoryMining
from tqdm import tqdm
from datetime import datetime
import time
import sys

commitMessage = []
commitDate = []
commitModification = []
repoSplit = []

def loadingBar(count,total,size):
    percent = float(count)/float(total)*100
    sys.stdout.write("\r" + str(int(count)).rjust(3,'0')+"/"+str(int(total)).rjust(3,'0') + ' [' + '='*int(percent/10)*size + ' '*(10-int(percent/10))*size + ']' + str(int(percent))+'%')

def recupFromRepo(s_repoPath, s_starting_date, s_ending_date, filename):
    s_date_format = "%d/%m/%Y"    
    d_starting_date = datetime.strptime(s_starting_date, s_date_format)
    d_ending_date = datetime.strptime(s_ending_date, s_date_format)
    
    commits = RepositoryMining(s_repoPath, None, d_starting_date, d_ending_date, None, None, None, None, None, "master").traverse_commits()
    
    nbCommits = len(list(commits))
    
    numberCommit = 0

    file = open(filename, "w",encoding="utf8")

    print("Number of commits : {}".format(nbCommits))

    #s_starting_date = "01/01/2020"
    #s_ending_date = "27/11/2020"

    start = time.time()

    s_result = ""

    #file.write("date\tmodified files\ttitle\tdetail\tmain branch\n")

    for commit in RepositoryMining(s_repoPath, None, d_starting_date, d_ending_date, None, None, None, None, None, "master").traverse_commits() :
        
        #file.write('Message {} , date {} , includes {} modified files'.format(commit.msg, commit.committer_date, len(commit.modifications)))
        date = str(commit.committer_date)
        date = date.split(" ")[0]
        commitDate.append(date)
        msg = commit.msg
        if commit.in_main_branch :
            mainBranch = 1
        else :
            mainBranch = 0

        msg = msg.split('\n')
        title = msg[0]
        if(len(msg) > 1):
            detail = msg[1]
        else :
            detail = ""

        nbModification = len(commit.modifications)
        commitModification.append(nbModification)

        file.write("{}\t{}\t{}\t{}\t{}\n".format(date, nbModification, title, detail ,mainBranch))

        #loadingBar(numberCommit,nbCommits,3)
        numberCommit += 1
        
    file.close()

    end = time.time()
    print("")
    print("time : {}".format(end - start))
    
def decoupeResult(file):
    with open(file) as fp:
       line = fp.readline()
       cnt = 1
       while line:
           #print(line)
           repoSplit.append(line.split(" , "))
           line = fp.readline()
           cnt += 1

def main():
    s_repoPath = input("path : ")
    #s_starting_date = input("starting date : ")
    #s_ending_date = input("ending date : ")
    #s_keyword = input("keyword : ")
    date = input("date : ")
    
    filename = str(date)+".txt"

    s_starting_date = "01/01/"+str(date)
    s_ending_date = "31/12/"+str(date)

    print("start : {} ".format(s_starting_date))
    print("end : {}".format(s_ending_date))
    
    recupFromRepo(s_repoPath,s_starting_date,s_ending_date,filename)

    print("")
    
    print("")
    input("Press any touch to continue...")

main()
