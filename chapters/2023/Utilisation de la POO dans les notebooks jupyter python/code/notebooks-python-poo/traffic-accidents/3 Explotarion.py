import pandas as pd
filename = 'accident_3_server_token_user.tsv'
dir_ = '../../../data/database/output_ml/M1/'
df = pd.read_csv(dir_+filename, delimiter = "\t", quoting = 3 ) # Part1
del df['Unnamed: 0']
df.shape
count_tweet_by_username = df['user_name'].value_counts() # Show distribution of tweets by user
count_tweet_by_username[0:40]
export = df[df['user_name'] == 'BogotaTransito'][['id_tweet','text','created_at','user_name','user_location']]
count_tweet_by_username = export['user_name'].value_counts() # Show distribution of tweets by user
count_tweet_by_username[0:40]
export.shape
export = df[['id_tweet','text','created_at','user_name','user_location']]
export
export.to_csv(dir_+"export_only_BogotaTransito_"+filename,sep='\t')
pd.set_option('display.max_colwidth',-1)
#df[df['user_name'] != 'BogotaTransito'].keys()
df[['text','user_name']][4000:4080]
df[df['user_name'] == 'RedapBogota'][['text','user_name']][0:40]




