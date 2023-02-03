import pandas as pd
search_api = pd.read_csv("../../../data/database/server_token_search.tsv", delimiter = "\t", quoting = 3)
user_search_api = pd.read_csv("../../../data/database/server_token_user.tsv", delimiter = "\t", quoting = 3)
timeline_stream_api = pd.read_csv("../../../data/database/server_follow_timeline_user.tsv", delimiter = "\t", quoting = 3)
bogota_stream_api.to_csv("../../../data/v1/doc2vec/bogota_stream_api.tsv",sep='\t')
dataset_no_bogota = pd.concat([search_api, user_search_api, timeline_stream_api])
dataset_no_bogota = dataset_no_bogota[["text","id_tweet"]]
dataset_no_bogota = dataset_no_bogota.drop_duplicates(['id_tweet'],keep='first')
dataset_no_bogota = dataset_no_bogota.drop_duplicates(['text'],keep='first')
dataset_no_bogota = dataset_no_bogota[["text"]]
dataset_no_bogota.to_csv("data/v1/doc2vec/no_bogota.tsv",sep='\t')
bogota_stream_api = pd.read_csv("data/database/server_bogota.tsv", delimiter = "\t", quoting = 3)
bogota_stream_api = bogota_stream_api[["text","id_tweet"]]
bogota_stream_api = bogota_stream_api.drop_duplicates(['id_tweet'],keep='first')
bogota_stream_api = bogota_stream_api.drop_duplicates(['text'],keep='first')
bogota_stream_api = bogota_stream_api.sample(frac=1)
#n=572510 representa el 50% de la cantidad final del conjunto de datos propuesta 1
bogota_stream_api = bogota_stream_api.sample(n=572510)
bogota_stream_api = bogota_stream_api[["text"]]

#bogota_stream_api.to_csv("data/v1/doc2vec/bogota_stream_api.tsv",sep='\t')
dataset_propuesta1 = pd.concat([bogota_stream_api, dataset_no_bogota])
dataset_propuesta1 = dataset_propuesta1.sample(frac=1)

dataset_propuesta1.to_csv("data/v1/doc2vec/dataset_propuesta1.tsv",sep='\t')
no_bogota = pd.read_csv("data/v1/doc2vec/no_bogota.tsv", delimiter = "\t", quoting = 3)
del no_bogota['Unnamed: 0']

dataset_propuesta2_complete = pd.concat([no_bogota, bogota_stream_api])
dataset_propuesta2_complete = dataset_propuesta2_complete.sample(frac=1)
dataset_propuesta2_complete.to_csv("data/v1/doc2vec/dataset_propuesta2_complete.tsv",sep='\t')
