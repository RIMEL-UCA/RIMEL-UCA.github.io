import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Lire le fichier CSV
df = pd.read_csv("models.csv")

categories = df['Category'].unique()
palette = sns.color_palette("hsv", len(categories))
color_map = dict(zip(categories, palette))

plt.figure(figsize=(12, 6))
for category in categories:
    subset = df[df['Category'] == category].sort_values(by='Year')  # Trier par année pour un tracé logique

    plt.scatter(subset['Year'], subset['Accuracy'], color=color_map[category], label=category, s=100)

    plt.plot(subset['Year'], subset['Accuracy'], color=color_map[category], linestyle='-', alpha=0.6)


    for _, row in subset.iterrows():
        plt.text(row['Year'] + 0.1, row['Accuracy'], row['Model'], fontsize=9, alpha=0.8)


plt.title("Natural Language Inference on RTE", fontsize=16)
plt.xlabel("Année", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=14)
plt.xticks(df['Year'].unique(), fontsize=10)  
plt.legend(title="Catégorie", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)


plt.show()
