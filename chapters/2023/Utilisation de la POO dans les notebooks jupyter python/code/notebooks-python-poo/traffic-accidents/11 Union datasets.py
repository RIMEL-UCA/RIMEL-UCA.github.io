import pandas as pd
dir_ = '../../data/database/output_ml/M1/NER_extractor/'

file_bogota = 'norm_entities_accident_1_server_bogota'
df_bogota = pd.read_csv(dir_+file_bogota+'.tsv', delimiter = "\t", quoting = 3)
df_bogota['dataset'] = '1_server_bogota'
print(df_bogota.shape)

file_token = 'norm_entities_accident_2_server_token_search'
df_token = pd.read_csv(dir_+file_token+'.tsv', delimiter = "\t", quoting = 3)
df_token['dataset'] = '2_server_token_search'
print(df_token.shape)

file_token_user = 'norm_entities_accident_3_server_token_user'
df_token_user = pd.read_csv(dir_+file_token_user+'.tsv', delimiter = "\t", quoting = 3)
df_token_user['dataset'] = '3_server_token_user'
print(df_token_user.shape)

file_timeline = 'norm_entities_accident_4_server_follow_timeline_user'
df_timeline = pd.read_csv(dir_+file_timeline+'.tsv', delimiter = "\t", quoting = 3)
df_timeline['dataset'] = '4_server_follow_timeline_user'
print(df_timeline.shape)
dataset = pd.concat([df_bogota,df_token,df_token_user,df_timeline])
dataset.shape
dataset.info()
dataset.iloc[0]["created_at"]
dataset = dataset[(dataset['created_at'] >= '2018-10-01') & (dataset['created_at'] < '2019-08-01')]
dataset.shape
dataset['dataset'].value_counts()
dataset = dataset.drop_duplicates(subset=['id_tweet'], keep='first')
dataset.shape
print(dataset.shape)
dataset['dataset'].value_counts()
dataset['user_name'].value_counts()
dataset.to_csv(dir_+"accidents_tweets.tsv",sep='\t',index=False)
