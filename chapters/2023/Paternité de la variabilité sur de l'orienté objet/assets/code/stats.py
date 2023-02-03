import json
import sys

if len(sys.argv) == 2:
    print("Il faut le chemin vers le fichier de résultat détaillé + général")
    exit(0)

result = json.loads(open(sys.argv[1]).read())
result_details = json.loads(open(sys.argv[2]).read())
print("Variants:", len(result_details))
print("VP:", len(result))
