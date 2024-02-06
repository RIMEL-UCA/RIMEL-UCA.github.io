import seaborn as sns
from matplotlib import pyplot as plt
from plotter.plotter_commits import *
from plotter.plotter_comments import *
from analysis.utils_analysis import choose_project
import matplotlib.dates as mdates

def plot_combined_data(first_file, second_file):
    # Read data from both files
    first_data = pd.read_csv(first_file)
    second_data = pd.read_csv(second_file)

    # Convert 'Period' column to datetime
    first_data['Period'] = pd.to_datetime(first_data['Period'], errors='coerce')
    second_data['Period'] = pd.to_datetime(second_data['Period'], errors='coerce')

    # Merge data on 'Period'
    combined_data = pd.merge(first_data, second_data, on='Period', how='outer')

    # Sort data by period
    combined_data.sort_values('Period', inplace=True)

    # Create a plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Plot contributors data in blue
    sns.lineplot(data=combined_data, x='Period', y='Non-English Percentage', ax=ax1, color='#846E84') #changer Number of Contributor par Non-English Percentage
    ax1.set_ylabel('Non-English Percentage', color='#846E84')

    # Plot commit percentage data in red
    sns.lineplot(data=combined_data, x='Period', y='Conventional Commit Percentage', ax=ax2, color='#E8B5BB')
    ax2.set_ylabel('Conventional Commit Percentage', color='#E8B5BB')

    # Set x-axis to show both years and months
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    plt.title('Non-English Pourcentage and Conventional Commit Percentage Over Time')
    plt.tight_layout()
    plt.show()


def plot_data():
    results_path = choose_project()

    while True:
        print("\nWhat data do you want to plot?")
        print("1. Author statistics")
        print("2. Conventional commits statistics")
        print("3. Author-period statistics")
        print("4. Contributors over time")
        print("5. Non-english commits over time")
        print("6. Author comments statistics")
        print("7. Conventional comments statistics")
        print("8. Author-period comments statistics")
        print("9. Before / after contributing")
        print("10. Plotter together")
        print("11. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            plot_one_data_commits(os.path.join(results_path, 'author_stats.csv'), "author_stats")
        elif choice == '2':
            plot_one_data_commits(os.path.join(results_path, 'time_stats.csv'), "Conventional Commit Percentage")
        elif choice == '3':
            plot_one_data_commits(os.path.join(results_path, 'author_period_stats.csv'), "author_period_stats")
        elif choice == '4':
            plot_one_data_commits(os.path.join(results_path, 'contributors_by_period.csv'), "Number of Contributor")
        elif choice == '5':
            plot_one_data_commits(os.path.join(results_path, 'non_english.csv'), "Non-English Percentage")
        elif choice == '6':
            plot_one_data_comments(os.path.join(results_path, 'author_comments_stats.csv'), "author_stats")
        elif choice == '7':
            plot_one_data_comments(os.path.join(results_path, 'time_comments_stats.csv'), "Conventional Commit Percentage")
        elif choice == '8':
            plot_one_data_comments(os.path.join(results_path, 'author_period_comments_stats.csv'), "author_period_stats")
        elif choice == '9':
            date = input("Enter date of CONTRIBUTING.md (format: YYYY-MM): ")
            plot_one_data_commits(os.path.join(results_path, 'time_stats.csv'), "before_after", date=date)
        elif choice == '10':
            print("Plotting stats...")
            first_file = input("Enter first filename: ")
            second_file = input("Enter second filename: ")
            plot_combined_data(first_file, second_file)
        elif choice == '11':
            break
        else:
            print("Invalid choice. Please try again.")
