import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

def plot_one_data_comments(file_path, plot_type):
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return
    
    data = pd.read_csv(file_path)

    if plot_type == 'author_stats' or plot_type == 'author_period_stats':
        if plot_type == 'author_stats':
            data = data.head(30)
            plt_title = 'CC Comment Percentage by Author'
            data['xlabel'] = data['Author']
        else:
            data = data.groupby('Year-Month').apply(lambda x: x.nlargest(1, 'Total Comments')).reset_index(drop=True)
            data['Year-Month'] = pd.to_datetime(data['Year-Month'])
            plt_title = 'CC Comment Percentage by Period by best Author'
            data['xlabel'] = data['Author'] + ' ' + data['Year-Month'].dt.strftime('%Y-%m')

        # Calculate the number of non-conventional commits
        data['Non-Conventional Comment'] = data['Total Comments'] - data['Conventional Comments']
        plt.figure(figsize=(20, 6))

        # Plot bar chart
        plt.bar(data['xlabel'], data['Non-Conventional Comment'], width=0.8, label='Non-Conventional Comments', color='#C79DA9')
        plt.bar(data['xlabel'], data['Conventional Comments'], width=0.8, bottom=data['Non-Conventional Comment'], label='CC Commits', color='#90EE90')
        
        plt.xticks(rotation=90, verticalalignment='top', ha='right')
        plt.xlabel('Author')
        plt.ylabel('Number of Comments')
        plt.legend()
        plt.title(plt_title)

    else:
        data['Year-Month'] = pd.to_datetime(data['Year-Month'], errors='coerce')
        data = data.dropna(subset=['Year-Month'])
        data.sort_values('Year-Month', inplace=True)  # Sort data by period

        # Calculate the inverse percentage
        data['Non-Conventional Comment'] = data['Total Comments'] - data['Conventional Comments']
        plt.figure(figsize=(10, 6))

        # Plot stacked bars for Percentage and Inverse Percentage
        plt.bar(data['Year-Month'], data['Conventional Comments'], width=20, label='Conventional Comment', color='#8FB7B7')
        plt.bar(data['Year-Month'], data['Non-Conventional Comment'], width=20, bottom=data['Conventional Comments'],
                label='Non-Conventional Comment', color='#C79DA9')

        # Set x-axis to show both years and months
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        plt.xticks(rotation=45)
        plt.xlabel('Period')
        plt.ylabel('Number of Comment')
        plt.legend()
        plt.title('Stacked Histogram of Conventional Comment Over Time')

    plt.tight_layout()
    plt.show()