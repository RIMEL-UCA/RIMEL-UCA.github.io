from postal.expand import expand_address
import pandas as pd
import re
## Variables para importar modelos y demás
dir_ = "../../data/v1/NER/"

file = 'tweets_location.tsv' # Dataset

## Importando Dataset
dataset = pd.read_csv(dir_+file, delimiter = "\t", quoting = 3)
locations = []
for i in range(len(dataset)):    
    loc = []
    text = dataset.iloc[i]['location']
    loc.append(text)
    expansions = expand_address(text, 
                                roman_numerals=False, 
                                split_alpha_from_numeric=True, 
                                expand_numex=False,
                                languages=["es"])
    
    expansions = re.sub(r'\b(\w+)( \1\b)+', r'\1', expansions[-1])
    loc = loc + [expansions]
    locations.append(loc)
df = pd.DataFrame(locations)
df.rename(columns={0:'original',1:'total'}, inplace=True)
df
df.to_csv(dir_+"tweets_location_libpostal_v3.tsv",sep='\t', index=False)

