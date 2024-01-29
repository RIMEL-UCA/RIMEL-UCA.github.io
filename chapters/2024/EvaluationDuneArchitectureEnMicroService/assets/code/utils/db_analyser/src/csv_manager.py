import csv

class CSV_Manager():
    def __init__(self):
        print("CSV MANAGER")

    def write(self, data):
        """
            On veut ca dans data, exemple :
            {'cassandra1': 1, 'client-bd': 1, 'analysis-db': 1, 'product-db': 1, 'marketing-db-slave': 3, 'marketing-db-master': 2, 'stats-db': 1}
        """

        # Le nom du fichier CSV à créer
        nom_fichier_csv = './datas/DB_analyse/db_usage.csv'

        # Écrire le dictionnaire dans un fichier CSV
        with open(nom_fichier_csv, mode='w', newline='') as fichier_csv:
            writer = csv.writer(fichier_csv)

            # Écrire l'en-tête du fichier CSV (les clés du dictionnaire)
            writer.writerow(['BD service name', 'Count'])

            # Écrire chaque paire clé-valeur dans le fichier CSV
            for cle, valeur in data.items():
                writer.writerow([cle, valeur])

        print(f'Le fichier CSV "{nom_fichier_csv}" a été créé avec succès')