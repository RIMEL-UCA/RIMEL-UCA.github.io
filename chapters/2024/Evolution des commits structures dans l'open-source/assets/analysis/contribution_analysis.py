import pandas as pd
from datetime import datetime

def read_commit_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    for i in range(0, len(lines), 4):
        author_line = lines[i].strip().split(": ")
        date_line = lines[i + 1].strip().split(": ")[1]

        author = author_line[1]
        date = datetime.strptime(date_line, "%Y-%m-%dT%H:%M:%SZ")

        data.append({'Author': author, 'Date': date})

    return data

def generate_contributor_statistics(data):
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])

    df.set_index('Date', inplace=True)
    df['YearMonth'] = df.index.to_period('M')

    contributor_stats = df.groupby(['YearMonth', 'Author']).size().unstack(fill_value=0)
    contributor_stats.to_csv('contributor_statistics.csv')

if __name__ == "__main__":
    commit_data_path = 'path/to/your/commit_data.txt'
    commit_data = read_commit_data(commit_data_path)
    generate_contributor_statistics(commit_data)
