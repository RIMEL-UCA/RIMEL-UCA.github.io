from pydriller import RepositoryMining
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdt
    
def plot_algo_occurences(src_file):
    data = pd.read_excel(src_file)
    df = pd.DataFrame(data)
    df = df.sort_values('both',ascending=False)
    plt.bar(df['name'], df['both'], tick_label = df['name'], 
        width = 0.8)
    plt.xlabel('Algorithmes') 
    plt.ylabel('Occurences') 
    plt.title('Occurences des différents algorithmes dans les commits')
    plt.xticks(rotation=90)
    plt.gcf().subplots_adjust(bottom=0.30)
    plt.show()

    
dest_file = "../Data/algorithmes_occurences_commitmsg_filtered.xlsx"
plot_algo_occurences(dest_file)
