from pydriller import RepositoryMining
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdt
from datetime import datetime
from pandas.plotting import register_matplotlib_converters


def autolabel(rects):
    """For ploting. Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

        
def diff_parsed(diff):
    """
    Returns a dictionary with the added and deleted lines.
    The dictionary has 2 keys: "added" and "deleted", each containing the
    corresponding added or deleted lines. For both keys, the value is a
    list of Tuple (int, str), corresponding to (number of line in the file,
    actual line).
    :return: Dictionary
    """
    lines = diff.split('\n')
    modified_lines = {'added': [], 'deleted': []}

    count_deletions = 0
    count_additions = 0

    for line in lines:
        line = line.rstrip()
        count_deletions += 1
        count_additions += 1

        if line.startswith('@@'):
            count_deletions, count_additions = _get_line_numbers(line)

        if line.startswith('-'):
            modified_lines['deleted'].append((count_deletions, line[1:]))
            count_additions -= 1

        if line.startswith('+'):
            modified_lines['added'].append((count_additions, line[1:]))
            count_deletions -= 1

        if line == r'\ No newline at end of file':
            count_deletions -= 1
            count_additions -= 1

    return modified_lines


def _get_line_numbers(line):
    token = line.split(" ")
    numbers_old_file = token[1]
    numbers_new_file = token[2]
    delete_line_number = int(numbers_old_file.split(",")[0].replace("-", "")) - 1
    additions_line_number = int(numbers_new_file.split(",")[0]) - 1
    return delete_line_number, additions_line_number


def plot_doc_over_time():
    """
    Plot le pourcentage de documentation ajouté dans les fichiers python,
    au fil des années. Les commentaires sont comptés comme de la documentation
    """
    register_matplotlib_converters()
    date = []
    totalAdd = []
    totalDelete = []
    totalLineAdded = []
    totalLineDeleted = []
    for commit in RepositoryMining('../../scikit-learn').traverse_commits():
        time = str(commit.author_date)
        year,month,day = time.split(" ")[0].split("-")
        year = int(year)
        inDoc = False
        countAdded = 0
        countDeleted = 0
        countLineAdded = 0
        countLineDeleted = 0
        for mod in commit.modifications:
            if(".py" in mod.filename):
                modification = diff_parsed(mod.diff)
                added = modification['added']
                deleted = modification['deleted']
                
                for line in added:
                    line = line[1]
                    countLineAdded +=1
                    if("#" in line and not inDoc):
                        countAdded+=1
                    elif("\"\"\"" in line):
                        if(inDoc):
                            countAdded+=1
                            inDoc=False
                        else:
                            inDoc=True
                    if(inDoc):
                        countAdded+=1
                
                inDoc = False
                for line in deleted:
                    line = line[1]
                    countLineDeleted +=1
                    if("#" in line and not inDoc):
                        countDeleted+=1
                    elif("\"\"\"" in line):
                        if(inDoc):
                            countDeleted+=1
                            inDoc=False
                        else:
                            inDoc=True
                    if(inDoc):
                        countDeleted+=1

        if(len(date) == 0):
            date.append(year)
            totalAdd.append(countAdded)
            totalDelete.append(countDeleted)
            totalLineAdded.append(countLineAdded)
            totalLineDeleted.append(countLineDeleted)
        elif(year in date):
            totalAdd[date.index(year)]+=countAdded
            totalDelete[date.index(year)]+=countDeleted
            totalLineAdded[date.index(year)]+= countLineAdded
            totalLineDeleted[date.index(year)]+= countLineDeleted
        elif(year not in date):
            date.append(year)
            totalAdd.append(countAdded)
            totalDelete.append(countDeleted)
            totalLineAdded.append(countLineAdded)
            totalLineDeleted.append(countLineDeleted)
        else:
            print("PROBLEME")
            
    newDoc = []
    newTotal = []
    for x in date:
       newDoc.append(totalAdd[date.index(x)] - totalDelete[date.index(x)])
       newTotal.append(totalLineAdded[date.index(x)] - totalLineDeleted[date.index(x)])
    pourcentDoc = []
    for z in date:
        p = newDoc[date.index(z)]
        t = newTotal[date.index(z)]
        pourcentDoc.append((p*100)/t)
    
    plt.title("Pourcentage d'ajout de documentation et commentaires au fil des années")
    plt.xlabel("Années")
    plt.ylabel("Pourcentage de documentation")
    plt.xticks(rotation=45)
    plt.tight_layout()
    filename= "pourcentage_doc_graph.png"
    path = "../Data/plots/" + filename
    plt.plot(date,pourcentDoc)
    plt.savefig(path)
    plt.clf()

    plt.title("Pourcentage d'ajout de documentation et de commentaires au fil des années")
    plt.xlabel("Années")
    plt.ylabel("Pourcentage de documentation")
    plt.xticks(rotation=45)
    plt.tight_layout()
    filename= "pourcentage_doc_barchart.png"
    path = "../Data/plots/" + filename
    plt.bar(date,pourcentDoc)
    plt.savefig(path)
    plt.clf()
              
    print(date)
    print(totalAdd)
    print(totalDelete)
    print(totalLineAdded)
    print(totalLineDeleted)
    print(newDoc)
    print(newTotal)
    print(pourcentDoc)


def plot_doc_and_comm(): 
    register_matplotlib_converters()
    date = []

    totalLineAdded = []
    totalLineDeleted = []
    
    totalAdded = []
    totalDeleted = []
    
    totalDocAdded = []
    totalDocDeleted = []
    
    totalCommAdded = []
    totalCommDeleted = []
    for commit in RepositoryMining('../../scikit-learn').traverse_commits():
        time = str(commit.author_date)
        year,month,day = time.split(" ")[0].split("-")
        year = int(year)
        inDoc = False
        countLineAdded = 0
        countLineDeleted = 0
        countTotalAdded = 0         #total de la doc ajoutée(commentaires + doc pure)
        countTotalDeleted = 0        #total de la doc supprimée(commentaires + doc pure)
        countDocAdded = 0           #total doc pure ajoutée (""" *doc* """)
        countDocDeleted = 0         #total doc pure supprimée (""" *doc* """)
        countCommAdded = 0          #total commentaires ajouté 
        countCommDeleted = 0           #total commentaires supprimé
        
        for mod in commit.modifications:
            if(".py" in mod.filename):               #focus sur les fichiers python
                modification = diff_parsed(mod.diff)
                added = modification['added']
                deleted = modification['deleted']
                
                for line in added:                                          #try not to puke challenge
                    countLineAdded+=1
                    if("#" in line and not inDoc and not "\"\"\"" in line):
                        countTotalAdded+=1
                        countCommAdded+=1
                    elif("#" in line and "\"\"\"" in line):
                        if(inDoc):
                            countTotalAdded+=1
                            countDocAdded+=1
                            inDoc=False
                        else:
                            if(line.index("#") < line.index("\"\"\"")):
                                countTotalAdded+=1
                                countCommAdded+=1
                            else:
                                inDoc=True
                    elif("\"\"\"" in line):
                        if(inDoc):
                            countTotalAdded+=1
                            countDocAdded+=1
                            inDoc=False
                        else:
                            inDoc=True
                    if(inDoc):
                        countTotalAdded+=1
                        countDocAdded+=1
                
                inDoc = False
                for line in deleted:
                    countLineDeleted+=1
                    if("#" in line and not inDoc and not "\"\"\"" in line):
                        countTotalDeleted+=1
                        countCommDeleted+=1
                    elif("#" in line and "\"\"\"" in line):
                        if(inDoc):
                            countTotalDeleted+=1
                            countDocDeleted+=1
                            inDoc=False
                        else:
                            if(line.index("#") < line.index("\"\"\"")):
                                countTotalDeleted+=1
                                countCommDeleted+=1
                            else:
                                inDoc=True
                    elif("\"\"\"" in line):
                        if(inDoc):
                            countTotalDeleted+=1
                            countDocDeleted+=1
                            inDoc=False
                        else:
                            inDoc=True
                    if(inDoc):
                        countTotalDeleted+=1
                        countDocDeleted+=1

        if(len(date) == 0):
            date.append(year)
            totalLineAdded.append(countLineAdded)
            totalLineDeleted.append(countLineDeleted)
            
            totalAdded.append(countTotalAdded)
            totalDeleted.append(countTotalDeleted)
            
            totalDocAdded.append(countDocAdded)
            totalDocDeleted.append(countDocDeleted)

            totalCommAdded.append(countCommAdded)
            totalCommDeleted.append(countCommDeleted)
            
        elif(year in date):
            totalLineAdded[date.index(year)]+= countLineAdded
            totalLineDeleted[date.index(year)]+= countLineDeleted
            
            totalAdded[date.index(year)]+= countTotalAdded
            totalDeleted[date.index(year)]+= countTotalDeleted
            
            totalDocAdded[date.index(year)]+= countDocAdded
            totalDocDeleted[date.index(year)]+= countDocDeleted

            totalCommAdded[date.index(year)]+= countCommAdded
            totalCommDeleted[date.index(year)]+= countCommDeleted
            
        elif(year not in date):
            date.append(year)
            totalLineAdded.append(countLineAdded)
            totalLineDeleted.append(countLineDeleted)
            
            totalAdded.append(countTotalAdded)
            totalDeleted.append(countTotalDeleted)
            
            totalDocAdded.append(countDocAdded)
            totalDocDeleted.append(countDocDeleted)

            totalCommAdded.append(countCommAdded)
            totalCommDeleted.append(countCommDeleted)
        else:
            print("PROBLEME")
            
    newTotalDoc = []
    newTotalLine = []
    newComm = []
    newDoc = []
    for x in date:
       newTotalLine.append(totalLineAdded[date.index(x)] - totalLineDeleted[date.index(x)])
       newTotalDoc.append(totalAdded[date.index(x)] - totalDeleted[date.index(x)])
       newDoc.append(totalDocAdded[date.index(x)] - totalDocDeleted[date.index(x)])
       newComm.append(totalCommAdded[date.index(x)] - totalCommDeleted[date.index(x)])
       
    pourcentDoc = []
    pourcentCommInDoc = []
    pourcentDocPureInDoc = []
    for z in date:
        dd = newTotalDoc[date.index(z)]
        dt = newTotalLine[date.index(z)]
        pourcentDoc.append((dd*100)/dt)

        dc = newComm[date.index(z)]
        pourcentCommInDoc.append((dc*100)/dd)

        d= newDoc[date.index(z)]
        pourcentDocPureInDoc.append((d*100)/dd)
    
    print(date)
    print(totalLineAdded)
    print(totalLineDeleted)
    print("\n")
    print("Total Added and Deleted")
    print(totalAdded)
    print(totalDeleted)
    print("\n")
    print("Total Documentation pure Added and Deleted")
    print(totalDocAdded)
    print(totalDocDeleted)
    print("\n")
    print("Total Comment Added and Deleted")
    print(totalCommAdded)
    print(totalCommDeleted)
    print("\n")
    print("Total Line")
    print(newTotalLine)
    print("Total Documentation")
    print(newTotalDoc)
    print("Total Comment")
    print(newComm)
    print("Total DocPure")
    print(newDoc)
    print("\n")
    print("Pourcent Doc")
    print(pourcentDoc)
    print("Pourcent Comment")
    print(pourcentCommInDoc)
    print("Pourcent Doc Pure")
    print(pourcentDocPureInDoc)

    #Plot meeeeee
    plt.title("Pourcentage d'ajout de documentation et de commentaires au fil des années")
    plt.xlabel("Années")
    plt.ylabel("Pourcentage de documentation")
    plt.xticks(rotation=45)
    plt.tight_layout()
    filename= "pourcentage_doc_barchart.png"
    path = "../Data/plots/" + filename
    plt.bar(date,pourcentDoc)
    plt.savefig(path)
    plt.clf()

    plt.title("Pourcentage de commentaires et de documentation pure dans la documentation")
    plt.xlabel("Années")
    plt.ylabel("Pourcentage total de documentation")
    plt.xticks(rotation=45)
    plt.tight_layout()
    filename= "pourcentage_com_doc_stackbarchart.png"
    path = "../Data/plots/" + filename
    p1 = plt.bar(date, pourcentDocPureInDoc, 0.35)
    p2 = plt.bar(date, pourcentCommInDoc, 0.35,bottom=pourcentDocPureInDoc)
    plt.legend((p1[0], p2[0]), ('Commentaires', 'Documentation Pure'))
    plt.savefig(path)
    plt.clf()


    plt.xticks(rotation=45)
    filename= "pourcentage_com_doc_doublebarchart.png"
    path = "../Data/plots/" + filename
    x = np.arange(len(date))  # the label locations
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 1/2, pourcentCommInDoc, 0.35, label='Commentaires')
    rects2 = ax.bar(x + 1/2, pourcentDocPureInDoc, 0.35, label='Documentation pure')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Années')
    ax.set_ylabel('Pourcentage total de documentation')
    ax.set_title('Pourcentage de commentaires et de documentation pure dans la documentation')
    ax.legend()
    fig.tight_layout()
    plt.savefig(path)
    plt.clf()

plot_doc_and_comm()
