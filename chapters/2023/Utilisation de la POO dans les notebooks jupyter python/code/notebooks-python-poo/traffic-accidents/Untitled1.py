from os import walk
import pandas as pd
dirname = "../../../data/v1/NER/brat/test/"
data = []
for (dirpath, dirnames, filenames) in walk(dirname):
    for filename in filenames:
        if filename.split(".")[1] == 'txt':        
            with open(dirname+filename, "r") as file:
                data.append(file.readlines()[0])
            i += 1
df = pd.DataFrame(data,columns=['tweet'])
df
df.to_csv("../../../data/v1/NER/ner_testset.tsv",sep='\t')
