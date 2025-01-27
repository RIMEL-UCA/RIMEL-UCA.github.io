import re
import matplotlib.pyplot as plt

def extract_commit_types(file_path):
    """Lit un fichier et extrait les types de commits avec leurs occurrences."""
    commit_types = {}
    commit_type_pattern = re.compile(r"- (\w+): (\d+)")
    in_commit_types_section = False  # Indique si on est dans la section "Types de commits"

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            # Activer la lecture uniquement après la ligne "Types de commits:"
            if line.strip() == "Types de commits:":
                in_commit_types_section = True
                continue

            # Arrêter la lecture si une autre section commence
            if in_commit_types_section and line.strip() == "":
                break

            # Extraire les types de commits si on est dans la bonne section
            if in_commit_types_section:
                match = commit_type_pattern.search(line)
                if match:
                    commit_type = match.group(1)
                    count = int(match.group(2))
                    commit_types[commit_type] = count

    return commit_types

def plot_commit_pie_chart(commit_types):
    # Trier les types de commits par ordre croissant de leur valeur
    sorted_commit_types = sorted(commit_types.items(), key=lambda item: item[1])

    labels, sizes = zip(*sorted_commit_types)
    colors = plt.cm.Paired(range(len(commit_types)))

    plt.figure(figsize=(8, 8))
    wedges = plt.pie(
        sizes, 
        labels=None,   
        autopct=None,  
        colors=colors, 
        startangle=140
    )

    # Légende avec les types de commits et leurs occurrences
    legend_labels = [f"{label}: {size}" for label, size in zip(labels, sizes)]
    plt.legend(legend_labels, title="Types de commits", loc="best", fontsize=10)


    plt.title("Répartition des types de commits", fontsize=14)
    plt.tight_layout()
    plt.savefig("commit_pie_chart.png")
    print("Le graphique trié a été sauvegardé sous le nom 'commit_pie_chart.png'")

if __name__ == "__main__":
    file_path = "commit_stats.txt"
    commit_types = extract_commit_types(file_path)
    plot_commit_pie_chart(commit_types)
