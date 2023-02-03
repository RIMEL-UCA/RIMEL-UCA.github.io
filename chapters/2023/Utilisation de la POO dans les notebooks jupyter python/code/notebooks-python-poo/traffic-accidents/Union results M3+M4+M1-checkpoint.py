import pandas as pd
filename = '1_server_bogota_part4.tsv'
dir_ = '../../../data/database/'
#df = pd.read_csv(dir_+filename, delimiter = "\t", quoting = 3 )
df = pd.read_csv(dir_+'server_bogota/'+filename, delimiter = "\t", quoting = 3 )
dfm1 = pd.read_csv(dir_+'output_ml/M1/clf_'+filename, delimiter = "\t", quoting = 3 ) # DBOW
dfm3 = pd.read_csv(dir_+'output_ml/M3/clf_'+filename, delimiter = "\t", quoting = 3) # 5_stem + SVM (Main)
dfm4 = pd.read_csv(dir_+'output_ml/M4/clf_'+filename, delimiter = "\t", quoting = 3) # 6_lemma + SVM

del dfm1['Unnamed: 0']
del dfm3['Unnamed: 0']
del dfm4['Unnamed: 0']

print(df.shape, dfm1.shape, dfm3.shape, dfm4.shape, sep='\n')
dfm1 = dfm1[['_id','label']]
dfm3 = dfm3[['_id','label']]
dfm4 = dfm4[['_id','label']]
print(dfm1.shape, dfm3.shape, dfm4.shape, sep='\n')
print(df['_id'].equals(dfm1['_id']))
print(df['_id'].equals(dfm3['_id']))
print(df['_id'].equals(dfm4['_id']))
if df['_id'].equals(dfm1['_id']) and df['_id'].equals(dfm3['_id']) and df['_id'].equals(dfm4['_id']):
    print('Processing...')
    df['label_m1'] = dfm1['label']
    df['label_m3'] = dfm3['label']
    df['label_m4'] = dfm4['label']
    print('Finish!')
else:
    print('Error: alguna de los dataset no son iguales')
df.loc[2223]
df.to_csv(dir_+"output_ml/clf_"+filename,sep='\t')
