from pydriller import RepositoryMining
import pandas as pd


data = pd.read_excel('../Data/types_algos_occurences.xlsx')
writer = pd.ExcelWriter('../Data/types.xlsx')
df = pd.DataFrame(data)
i=0
while i < len(df):
    cmp = 0
    string = df.at[i,'name'].lower()
    for commit in RepositoryMining('../../scikit-learn').traverse_commits():
        if (string in commit.msg.lower()):
            cmp+=1
    df.at[i,'nb']=cmp
    i+=1
df.to_excel(writer)
writer.save()
writer.close()
