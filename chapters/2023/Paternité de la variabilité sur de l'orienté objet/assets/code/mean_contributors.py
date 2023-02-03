import json
import sys
from functools import reduce

if len(sys.argv) == 1:
    print("Il faut le chemin vers le fichier de résultat")
    exit(0)

result = json.loads(open(sys.argv[1]).read())
sum_authors = reduce(lambda a, b: a+b, map(lambda authors: len(authors), result.values()))
sum_variants = len(result.keys())
mean_authors = sum_authors / sum_variants
print(mean_authors)

#  python3 ./mean_contributors.py results/FizzBuzzEnterpriseEdition_paternity_result_detail.txt
