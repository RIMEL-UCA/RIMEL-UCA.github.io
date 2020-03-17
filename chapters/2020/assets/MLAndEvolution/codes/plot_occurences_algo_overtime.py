from pydriller import RepositoryMining
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdt
from datetime import datetime
from pandas.plotting import register_matplotlib_converters


def algos_over_time(src_file): 
    register_matplotlib_converters()
    data = pd.read_excel(src_file)
    df = pd.DataFrame(data)
    i = 0
    loop = len(df.index)
    while i < loop:
        bigtable = []
        occu=[]
        date = []
        for commit in RepositoryMining('../../scikit-learn').traverse_commits():
            str1 = df.at[i,'name']
            if(not pd.isnull(df.at[i, 'abreviation'])):
                str2 = "" + df.at[i, 'abreviation']+ ""
                if((str1.lower() in commit.msg.lower()) or (str2.lower() in commit.msg.lower())):
                   time = str(commit.author_date)
                   year,month,day = time.split(" ")[0].split("-")
                   data = datetime(int(year),int(month),int(day))
                   bigtable.append(data)
            else:
                if(str1.lower() in commit.msg.lower()):
                   time = str(commit.author_date)
                   year,month,day = time.split(" ")[0].split("-")
                   data = datetime(int(year),int(month),int(day))
                   bigtable.append(data)
        if(bigtable):
            bigtable.sort()
            y = 1
            for x in bigtable:
                if len(date) == 0:
                    date.append(x)
                else:
                    if x.year == date[len(date)-1].year and x.month == date[len(date)-1].month:
                        y+=1
                    else:
                        occu.append(y)
                        y = 1
                        date.append(x)
            occu.append(y)    
        plt.title("Occurences of " + df.at[i,'plot_name'] +" algorithm over time")
        plt.xlabel("Time")
        plt.ylabel("Number of occurences")
        plt.xticks(rotation=45)
        filename= df.at[i,'plot_name'] + " occurences over time.png"
        path = "../Data/plots/occurences_over_time/" + filename
        plt.plot(date,occu)
        plt.savefig(path)
        plt.clf()
        i+=1

dest_file = "../Data/algorithmes_occurences_commitmsg_filtered_origin.xlsx"
algos_over_time(dest_file)

