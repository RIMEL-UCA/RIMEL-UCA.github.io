import pandas as pd
filename = 'accident_2_server_token_search.tsv'
dir_ = '../../../data/database/output_ml/M1/'
df = pd.read_csv(dir_+filename, delimiter = "\t", quoting = 3 ) # Part1
del df['Unnamed: 0']
df.shape
count_tweet_by_username = df['user_name'].value_counts() # Show distribution of tweets by user
count_tweet_by_username[0:40]
export = df[df['user_name'] != 'BogotaTransito'][['id_tweet','text','created_at','user_name','user_location']]
export = export[export['user_name'] != 'rutassitp']
export = export[export['user_name'] != 'WazeTrafficBOG']
export = export[export['user_name'] != 'RedapBogota']
export = export[export['user_name'] != 'Citytv']
export = export[export['user_name'] != 'CIVICOSBOG']
count_tweet_by_username = export['user_name'].value_counts() # Show distribution of tweets by user
count_tweet_by_username[0:40]
export.shape
export.to_csv(dir_+"export_"+filename,sep='\t')
pd.set_option('display.max_colwidth',-1)
df[df['user_name'] != 'BogotaTransito'].keys()
df[(df['user_name'] != 'MarthaECamargo') & (df['user_name'] != 'rutassitp')][['text','user_name']][180:220]

