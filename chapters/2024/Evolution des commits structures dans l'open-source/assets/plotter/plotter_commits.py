import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import os

def get_total_conv_commits(data):
    return data['Feat Commits'] + data['Release Commits'] \
                                + data['Fix Commits'] \
                                + data['Test Commits'] \
                                + data['Clean Commits'] \
                                + data['Doc Commits']

def plot_one_data_commits(file_path, plot_type, date=None):
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return
    
    data = pd.read_csv(file_path)

    if plot_type == 'author_stats' or plot_type == 'author_period_stats':
        if plot_type == 'author_stats':
            data = data.head(30)
            plt_title = 'Conventional Commit Percentage by Author'
            data['xlabel'] = data['Author']
        else:
            data = data.groupby('Period').apply(lambda x: x.nlargest(1, 'Total Commits')).reset_index(drop=True)
            data['Period'] = pd.to_datetime(data['Period'])
            plt_title = 'Conventional Commit Percentage by Period by best Author'
            data['xlabel'] = data['Author'] + ' ' + data['Period'].dt.strftime('%Y-%m')

        # Calculate the number of non-conventional commits
        data['Non-Conventional Commit'] = data['Total Commits'] - get_total_conv_commits(data)
        plt.figure(figsize=(20, 6))

        # Plot bar chart
        plt.bar(data['xlabel'], data['Non-Conventional Commit'], width=0.8, label='Non-Conventional Commits', color='#C79DA9')
        plt.bar(data['xlabel'], data['Feat Commits'], width=0.8, bottom=data['Non-Conventional Commit'], label='Feat Commits', color='#90EE90')
        plt.bar(data['xlabel'], data['Release Commits'], width=0.8, bottom=data['Non-Conventional Commit'] + data['Feat Commits'], label='Release Commits', color='#32CD32')
        plt.bar(data['xlabel'], data['Fix Commits'], width=0.8, bottom=data['Non-Conventional Commit'] + data['Feat Commits'] + data['Release Commits'], label='Fix Commits', color='#228B22')
        plt.bar(data['xlabel'], data['Test Commits'], width=0.8, bottom=data['Non-Conventional Commit'] + data['Feat Commits'] + data['Release Commits'] + data['Fix Commits'], label='Test Commits', color='#808000')
        plt.bar(data['xlabel'], data['Clean Commits'], width=0.8, bottom=data['Non-Conventional Commit'] + data['Feat Commits'] + data['Release Commits'] + data['Fix Commits'] + data['Test Commits'], label='Clean Commits', color='#006400')
        plt.bar(data['xlabel'], data['Doc Commits'], width=0.8, bottom=data['Non-Conventional Commit'] + data['Feat Commits'] + data['Release Commits'] + data['Fix Commits'] + data['Test Commits'] + data['Clean Commits'], label='Doc Commits', color='#2E8B57')
        plt.bar(data['xlabel'], data['Other Commits'], width=0.8, bottom=data['Non-Conventional Commit'] + data['Feat Commits'] + data['Release Commits'] + data['Fix Commits'] + data['Test Commits'] + data['Clean Commits'] + data['Doc Commits'], label='Refactor Commits', color='#00FA9A')

        plt.xticks(rotation=90, verticalalignment='top', ha='right')
        plt.xlabel('Author')
        plt.ylabel('Number of Commits')
        plt.legend()
        plt.title(plt_title)

    elif plot_type == 'before_after' and date is not None:
        date = pd.to_datetime(date)
        data['Period'] = pd.to_datetime(data['Period'], errors='coerce')
        
        # Filter data before and after date
        before_data = data[data['Period'] < date]
        after_data = data[data['Period'] >= date]

        # Calculate the number of non-conventional commits
        before_data['Non-Conventional Commit'] = before_data['All Commit'] - before_data['Conventional Commit']
        after_data['Non-Conventional Commit'] = after_data['All Commit'] - after_data['Conventional Commit']

        # Get total conventional commits and non-conventional commits
        before_total_data = before_data.groupby('Period').sum().reset_index()
        before_total = before_total_data['All Commit'].sum()
        before_conventional = before_total_data['Conventional Commit'].sum()
        before_non_conventional = before_total_data['Non-Conventional Commit'].sum()
        
        after_total_data = after_data.groupby('Period').sum().reset_index()
        after_total = after_total_data['All Commit'].sum()
        after_conventional = after_total_data['Conventional Commit'].sum()
        after_non_conventional = after_total_data['Non-Conventional Commit'].sum()

        print(before_conventional / before_total, before_non_conventional / before_total)
        print(after_conventional / after_total, after_non_conventional / after_total)

        # Define labels, values, and colors
        labels = ['Before {}'.format(date.date()), 'After {}'.format(date.date())]
        conventional_values = [before_conventional, after_conventional]
        non_conventional_values = [before_non_conventional, after_non_conventional]
        colors = ['#90EE90', '#C79DA9']

        # Plot bar chart
        plt.figure(figsize=(20, 6))
        plt.bar(labels, conventional_values, color=colors[0], label='Conventional Commits')
        plt.bar(labels, non_conventional_values, bottom=conventional_values, color=colors[1], label='Non-Conventional Commits')
        plt.axvline(x=0.5, color='r', linestyle='--', label='Date: {}'.format(date.date()))
        plt.legend()
        plt.title('Before / After CONTRIBUTING.md Conventional Commit Percentage')

    else:
        data['Period'] = pd.to_datetime(data['Period'], errors='coerce')
        data = data.dropna(subset=['Period'])
        data.sort_values('Period', inplace=True)  # Sort data by period

        # Calculate the inverse percentage
        data['Non-Conventional Commit'] = data['All Commit'] - data['Conventional Commit']
        plt.figure(figsize=(10, 6))

        # Plot stacked bars for Percentage and Inverse Percentage
        plt.bar(data['Period'], data['Conventional Commit'], width=20, label='Conventional Commit', color='#8FB7B7')
        plt.bar(data['Period'], data['Non-Conventional Commit'], width=20, bottom=data['Conventional Commit'],
                label='Non-Conventional Commit', color='#C79DA9')

        # Set x-axis to show both years and months
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        plt.xticks(rotation=45)
        plt.xlabel('Period')
        plt.ylabel('Number of Commit')
        plt.legend()
        plt.title('Stacked Histogram of Conventional Commit Over Time')

    plt.tight_layout()
    plt.show()