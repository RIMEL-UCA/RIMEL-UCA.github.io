import matplotlib.pyplot as plt
import json

# Load the data for the repositories that use OOP
repositories_forks_class = {}
with open('repositories_forks_class.txt', 'r') as f:
    repositories_forks_class = json.load(f)
repositories_oop = list(repositories_forks_class.values())

# Load the data for the repositories that don't use OOP
repositories_forks_no_class = {}
with open('repositories_forks_no_class.txt', 'r') as f:
    repositories_forks_no_class = json.load(f)
repositories_no_oop = list(repositories_forks_no_class.values())

# Plot the histograms for the two datasets
plt.hist(repositories_oop, bins=20, alpha=0.5, label='OOP')
plt.hist(repositories_no_oop, bins=20, alpha=0.5, label='Not OOP')

# Add labels and title
plt.xlabel('Number of forks')
plt.ylabel('Frequency')
plt.title('Comparison of Fork Distributions for no Using OOP vs Not Using OOP')
plt.legend()

# Show the plot
plt.show()
