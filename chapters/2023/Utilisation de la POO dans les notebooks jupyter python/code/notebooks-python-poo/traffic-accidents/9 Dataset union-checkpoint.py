import pandas as pd
dir_ = "../../data/database/output_ml/M1/"
file = 'accident_1_server_bogota_part1.tsv'
dataset = pd.read_csv(dir_+file, delimiter = "\t", quoting = 3)
dataset.info()

