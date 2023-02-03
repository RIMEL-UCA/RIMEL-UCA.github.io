from functools import reduce

import matplotlib.pyplot as plt
import json
import sys
from itertools import chain

if len(sys.argv) == 1:
    print("Il faut le chemin vers le fichier de résultat")
    exit(0)

plt.rc('xtick', labelsize=4)
result = json.loads(open(sys.argv[1]).read())
variants_count = len(result.keys())
authors = set(chain(*result.values()))
authors_contribution = dict()
for author in authors:
    # print(result.items())
    contribution = reduce(lambda a, b: a + b, map(lambda authors_var: authors_var[author],
                                                  filter(lambda authors_var: author in authors_var,
                                                              result.values())))
    authors_contribution[author] = (contribution / variants_count) * 100

fig, ax = plt.subplots()

ax.bar(list(authors), authors_contribution.values(), align='edge')
plt.xticks(rotation=45)
ax.set_ylabel('Pourcentage de contribution')
ax.set_title('Pourcentage de contribution pour chaque auteurs')

# print(sum(authors_contribution.values())) ->

print("Nombre de développeurs:", len(authors))
plt.show()

# python3 ./percentage_contribution_mean.py results/FizzBuzzEnterpriseEdition_paternity_result_detail.txt
