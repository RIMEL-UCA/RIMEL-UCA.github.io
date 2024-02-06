import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    # Example usage
    print("Plotting stats...")
    first_file = input("Enter first filename: ")
    second_file = input("Enter second filename: ")
    plot_combined_data(first_file, second_file)