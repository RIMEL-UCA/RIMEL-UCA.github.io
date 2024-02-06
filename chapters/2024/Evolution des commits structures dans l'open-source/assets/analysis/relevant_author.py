import pandas as pd

def find_active_authors_in_months(file_path, months):
    data = pd.read_csv(file_path)
    data['Period'] = pd.to_datetime(data['Period'])

    active_authors = set()
    for month in months:
        month_data = data[data['Period'] == month]
        active_authors.update(month_data['Author'])

    return active_authors
    
# Example Usage
file_path = './author_period_stats.csv'
months = ['2014-08','2015-12', '2017-11', '2019-9']  # Add your months here in 'YYYY-MM' format
active_authors = find_active_authors_in_months(file_path, months)
print("Active authors in the given months:", active_authors)

