import os

from scrapping import comment_scraping, commit_scraping
from analysis import commit_analysis, comment_analysis
import plotter.plotter as plotter

def main():
    while True:
        print("\nMenu principal")
        print("1. Scrapper les données des commits")
        print("2. Scrapper les données des comments (issues + pull requests + reviews)")
        print("3. Analyser les données des commits existants")
        print("4. Analyser les données des comments existants")
        print("5. Plot les donnees csv")
        print("6. Quitter")
        choice = input("Entrez votre choix : ")

        if choice == '1':
            commit_scraping.scrape_data()
        elif choice == '2':
            comment_scraping.scrape_data()
        elif choice == '3':
            if os.path.isdir('results'):
                commit_analysis.perform_analysis()
            else:
                print("Aucun fichier de données trouvé. Veuillez d'abord scrapper les données.")
        elif choice == '4':
            if os.path.isdir('results'):
                comment_analysis.perform_analysis()
            else:
                print("Aucun fichier de données trouvé. Veuillez d'abord scrapper les données.")
        elif choice == '5':
            plotter.plot_data()
        elif choice == '6':
            print("Fin du programme.")
            break
        else:
            print("Choix invalide. Veuillez réessayer.")

if __name__ == "__main__":
    main()
