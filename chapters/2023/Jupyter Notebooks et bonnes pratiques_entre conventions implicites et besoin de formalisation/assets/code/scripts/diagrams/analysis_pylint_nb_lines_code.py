import glob
import json
import tomli
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def plot_profile(source: str):
    elems = dict()
    toml_files = list(glob.glob(('../notebooks/' + source + '/*/*.toml'), recursive=True))
    for toml_file in toml_files:
        with open(toml_file, "rb") as f2:
            data = tomli.load(f2)
            if data['metadata']['date'].lower() == 'todo':
                print(f'Skipping {toml_file} because of invalid todo date')
                continue
            if source == 'github':
                convert_date = datetime.strptime(data['metadata']['date'], '%d/%m/%Y').date()
            else:
                convert_date = datetime.strptime(data['metadata']['date'], '%Y-%m-%d').date()

            elems[data['title']] = convert_date
    print(len(elems))
    json_results = list(glob.glob('../results/*.json', recursive=True))
    x = []
    y = []
    for key, val in elems.items():
        res = [i for i in json_results if key in i]
        if len(res) > 0:
            with open(res[0]) as f:
                json_data = json.load(f)
                if "profile" in json_data:
                    nb_lines_code = 0
                    for p in json_data['profile']:
                        if p['cell_type'] == 'code':
                            nb_lines_code += p['nb_lines']
                    x.append(val)
                    y.append(nb_lines_code)
                    print(str(nb_lines_code) + " " + key)
    print(len(y))
    return x, y


github_result = plot_profile('github')
kaggle_result = plot_profile('kaggle')
fig, ax = plt.subplots()

github_scatter = ax.scatter(github_result[0], github_result[1], linewidth=2.0, c='red')
kaggle_scatter = ax.scatter(kaggle_result[0], kaggle_result[1], linewidth=2.0, c='blue')
ax.legend((github_scatter, kaggle_scatter), ("GitHub", "Kaggle"))
ax.set_xlabel('date')
ax.set_ylabel('number of lines of code')
plt.xticks(rotation=70)
plt.show()
