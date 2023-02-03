import pandas as pd
filename = 'no_accident_1_server_bogota'
#dir_ = '../../../data/database/output_ml/M1/'
dir_ = '../../../data/database/output_ml/M2/'
#df = pd.read_csv(dir_+filename, delimiter = "\t", quoting = 3 )
dfp1 = pd.read_csv(dir_+filename+'_part1.tsv', delimiter = "\t", quoting = 3 ) # Part1
dfp2 = pd.read_csv(dir_+filename+'_part2.tsv', delimiter = "\t", quoting = 3 ) # Part2
dfp3 = pd.read_csv(dir_+filename+'_part3.tsv', delimiter = "\t", quoting = 3 ) # Part2
dfp4 = pd.read_csv(dir_+filename+'_part4.tsv', delimiter = "\t", quoting = 3 ) # Part2
df = pd.concat([dfp1,dfp2,dfp3, dfp4])

del dfp1['Unnamed: 0']
del dfp2['Unnamed: 0']
del dfp3['Unnamed: 0']
del dfp4['Unnamed: 0']
del df['Unnamed: 0']

print(df.shape,dfp1.shape, dfp2.shape, dfp3.shape, dfp4.shape, sep='\n')
df.to_csv(dir_+filename+".tsv",sep='\t')
