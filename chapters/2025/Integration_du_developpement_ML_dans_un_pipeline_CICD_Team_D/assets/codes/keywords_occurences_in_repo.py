import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

VALID_EXTENSIONS = [".txt", ".csv", ".log", ".py", ".sh", ".bat", ".md", ".js", ".html", ".css", ".json", ".xml"]

def search_keywords_in_file(file_path, keywords):
    occurrences = {keyword: [] for keyword in keywords}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line_lower = line.lower()
                for keyword in keywords:
                    if keyword.lower() in line_lower:
                        occurrences[keyword].append(line_number)
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier {file_path}: {e}")
    return {k: v for k, v in occurrences.items() if v}

def analyze_directory(directory, keywords):
    results = {keyword: {"count": 0, "files": []} for keyword in keywords}

    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in VALID_EXTENSIONS):
                file_path = os.path.join(root, file)
                file_occurrences = search_keywords_in_file(file_path, keywords)
                for keyword, occurrences in file_occurrences.items():
                    results[keyword]["count"] += len(occurrences)
                    results[keyword]["files"].append((file_path, occurrences))

    return results

def display_results(results, output_file=None):
    output_lines = []
    for keyword, data in results.items():
        if data["count"] > 0:
            output_lines.append(f"Mot-clé: '{keyword}'")
            output_lines.append(f"  Nombre d'occurrences: {data['count']}")
            output_lines.append(f"  Apparaît dans les fichiers:")
            for file, occurrences in data["files"]:
                output_lines.append(f"    - {file}")
                output_lines.append(f"      Lignes: {', '.join(map(str, occurrences))}")
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
    else:
        print("\n".join(output_lines))

def plot_keyword_heatmap(results, base_directory, output_file=None):
    directories = {}
    keywords = list(results.keys())

    for keyword in keywords:
        for file, occurrences in results[keyword]["files"]:
            directory = os.path.dirname(file)
            relative_directory = os.path.relpath(directory, base_directory)
            if relative_directory not in directories:
                directories[relative_directory] = {kw: 0 for kw in keywords}
            directories[relative_directory][keyword] += len(occurrences)

    dir_list = list(directories.keys())
    
    heatmap_data = np.zeros((len(dir_list), len(keywords)))
    for dir_idx, directory in enumerate(dir_list):
        for keyword_idx, keyword in enumerate(keywords):
            heatmap_data[dir_idx, keyword_idx] = directories[directory][keyword]

    keyword_sums = np.sum(heatmap_data, axis=0)
    active_keywords_indices = np.where(keyword_sums > 0)[0]
    filtered_keywords = [keywords[i] for i in active_keywords_indices]
    filtered_heatmap_data = heatmap_data[:, active_keywords_indices]

    if len(filtered_keywords) > 0:
        plt.figure(figsize=(10, len(dir_list) * 0.5))
        sns.heatmap(filtered_heatmap_data, 
                   annot=True, 
                   fmt="g", 
                   xticklabels=filtered_keywords, 
                   yticklabels=dir_list, 
                   cmap="YlGnBu")
        plt.xlabel('Keywords')
        plt.ylabel('Directories')
        plt.title('Keyword Occurrences Heatmap')

        if output_file:
            plt.savefig(output_file, bbox_inches="tight")
        else:
            plt.show()
    else:
        print("No keywords with occurrences found for the heatmap visualization.")

def main():

    if len(sys.argv) < 3:
        print("Usage: python keywords_occurences_in_repo.py <directory> <keyword1> <keyword2> ...")
        return

    directory = sys.argv[1]
    keywords = sys.argv[2:]

    if not os.path.isdir(directory):
        print("Le chemin spécifié n'est pas un dossier valide.")
        return

    console_output_file = "output.txt"
    heatmap_output_file = "heatmap.png"

    print("Analyse en cours du répertoire:", directory)
    results = analyze_directory(directory, keywords)

    if any(data["count"] > 0 for data in results.values()):
        print("\nRésultats de l'analyse:")
        display_results(results, output_file=console_output_file)
        print(f"\nLes résultats de la console ont été sauvegardés dans: {console_output_file}")

        plot_keyword_heatmap(results, directory, output_file=heatmap_output_file)
        print(f"\nLa heatmap a été sauvegardée dans: {heatmap_output_file}")
    else:
        print("Aucune occurrence trouvée pour les mots-clés spécifiés.")

if __name__ == "__main__":
    main()
