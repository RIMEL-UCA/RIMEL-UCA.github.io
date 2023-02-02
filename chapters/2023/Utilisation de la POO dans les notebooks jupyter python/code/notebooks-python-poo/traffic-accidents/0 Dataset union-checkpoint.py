import pandas as pd
dir_ = "../../../data/database/output_ml/M1/"
file = 'accident_1_server_bogota_part'

df1 = pd.read_csv(dir_+file+'1.tsv', delimiter = "\t", quoting = 3)
del df1['Unnamed: 0']

df2 = pd.read_csv(dir_+file+'2.tsv', delimiter = "\t", quoting = 3)
del df2['Unnamed: 0']

df3 = pd.read_csv(dir_+file+'3.tsv', delimiter = "\t", quoting = 3)
del df3['Unnamed: 0']

df4 = pd.read_csv(dir_+file+'4.tsv', delimiter = "\t", quoting = 3)
del df4['Unnamed: 0']
print(df1.shape)
print(df2.shape)
print(df3.shape)
print(df4.shape)
dataset = pd.concat([df1,df2,df3,df4])
dataset.shape
dataset
