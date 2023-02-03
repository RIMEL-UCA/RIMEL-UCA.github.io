import pandas as pd
dir_ = '../../data/database/output_ml/M1/NER_extractor/'

file_bogota = 'accidents_tweets'
dataset = pd.read_csv(dir_+file_bogota+'.tsv', delimiter = "\t", quoting = 3)
dataset.shape
dataset.head(5)
df = dataset[(dataset['created_at'] >= '2018-10-01') & (dataset['created_at'] < '2018-11-01')]
print("Min:", min(df['created_at']))
print("MAx:", max(df['created_at']))
print(df.shape)
df['dataset'].value_counts()
df['user_name'].value_counts()
df = dataset[(dataset['created_at'] >= '2018-11-01') & (dataset['created_at'] < '2018-12-01')]
print("Min:", min(df['created_at']))
print("MAx:", max(df['created_at']))
print(df.shape)
### Tweets por Dataset
df['dataset'].value_counts()
### #Tweets por username
df['user_name'].value_counts()
df = dataset[(dataset['created_at'] >= '2018-12-01') & (dataset['created_at'] < '2019-01-01')]
print("Min:", min(df['created_at']))
print("MAx:", max(df['created_at']))
print(df.shape)
### Tweets por Dataset
df['dataset'].value_counts()
### #Tweets por username
df['user_name'].value_counts()
df = dataset[(dataset['created_at'] >= '2019-01-01') & (dataset['created_at'] < '2019-02-01')]
print("Min:", min(df['created_at']))
print("MAx:", max(df['created_at']))
print(df.shape)
### Tweets por Dataset
df['dataset'].value_counts()
### #Tweets por username
df['user_name'].value_counts()
df = dataset[(dataset['created_at'] >= '2019-02-01') & (dataset['created_at'] < '2019-03-01')]
print("Min:", min(df['created_at']))
print("MAx:", max(df['created_at']))
print(df.shape)
### Tweets por Dataset
df['dataset'].value_counts()
### #Tweets por username
df['user_name'].value_counts()
df = dataset[(dataset['created_at'] >= '2019-03-01') & (dataset['created_at'] < '2019-04-01')]
print("Min:", min(df['created_at']))
print("MAx:", max(df['created_at']))
print(df.shape)
### Tweets por Dataset
df['dataset'].value_counts()
### #Tweets por username
df['user_name'].value_counts()
df = dataset[(dataset['created_at'] >= '2019-04-01') & (dataset['created_at'] < '2019-05-01')]
print("Min:", min(df['created_at']))
print("MAx:", max(df['created_at']))
print(df.shape)
### Tweets por Dataset
df['dataset'].value_counts()
### #Tweets por username
df['user_name'].value_counts()
df = dataset[(dataset['created_at'] >= '2019-05-01') & (dataset['created_at'] < '2019-06-01')]
print("Min:", min(df['created_at']))
print("MAx:", max(df['created_at']))
print(df.shape)
### Tweets por Dataset
df['dataset'].value_counts()
### #Tweets por username
df['user_name'].value_counts()
df = dataset[(dataset['created_at'] >= '2019-06-01') & (dataset['created_at'] < '2019-07-01')]
print("Min:", min(df['created_at']))
print("MAx:", max(df['created_at']))
print(df.shape)
### Tweets por Dataset
df['dataset'].value_counts()
### #Tweets por username
df['user_name'].value_counts()
df = dataset[(dataset['created_at'] >= '2019-07-01') & (dataset['created_at'] < '2019-08-01')]
print("Min:", min(df['created_at']))
print("MAx:", max(df['created_at']))
print(df.shape)
### Tweets por Dataset
df['dataset'].value_counts()
### #Tweets por username
df['user_name'].value_counts()
df = dataset[(dataset['created_at'] >= '2019-08-01') & (dataset['created_at'] < '2019-09-01')]
print("Min:", min(df['created_at']))
print("MAx:", max(df['created_at']))
print(df.shape)
