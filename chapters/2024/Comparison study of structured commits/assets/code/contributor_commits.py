from pydriller import Repository
import pandas as pd
import matplotlib.pyplot as plt
import asyncio
from concurrent.futures import ThreadPoolExecutor

repos = [
    "https://github.com/Kanaries/pygwalker",
    "https://github.com/angular/angular",
    "https://github.com/nodejs/node",
    "https://github.com/tensorflow/tensorflow",
    #"https://github.com/facebook/react",
    "https://github.com/Netflix/Hystrix"
]

conventional_commit_keywords = [
    'feat', 'feature', 'fix', 'chore', 'docs', 'doc', 'style', 'refactor', 
    'perf', 'test', 'design', 'build', 'cleanup'
]

#check if a commit message is conventional
def is_conventional_commit(message):
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in conventional_commit_keywords)


# Process a repository
async def process_repo(repo_url, executor):
    repo = Repository(repo_url)
    commit_data = {
        'Author': [],
        'Conventional': [],
        'Message': []
    }

    non_conventional_messages = []

    # Process commits
    for commit in repo.traverse_commits():
        is_conventional = is_conventional_commit(commit.msg)
        commit_data['Author'].append(commit.author.name)
        commit_data['Conventional'].append(is_conventional)
        commit_data['Message'].append(commit.msg)
        if not is_conventional:
            non_conventional_messages.append(commit.msg)
    
    # Write non conventional commit messages to a file
    non_conventional_filename = f'{repo_url.split("/")[-1]}_non_conventional_commits.txt'
    with open(non_conventional_filename, 'w') as file:
        for message in non_conventional_messages:
            file.write(message + '\n\n')
    
    commit_df = pd.DataFrame(commit_data)

    # Calculate the conventional commit rate for each contributor
    author_stats = commit_df.groupby('Author')['Conventional'].agg(['count', 'sum'])
    author_stats['ConventionalRate'] = author_stats['sum'] / author_stats['count']
    top_contributors = author_stats.sort_values(by='count', ascending=False).head(5)

    # Plotting the top 5 contributors
    fig, ax = plt.subplots()
    top_contributors['ConventionalRate'].plot(kind='bar', ax=ax)
    ax.set_title('Top 5 Contributors Conventional Commit Rate')
    ax.set_xlabel('Contributor')
    ax.set_ylabel('Conventional Commit Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    filename = f'{repo_url.split("/")[-1]}_top_contributors_commit_rate.pdf'

    # Save the plot
    plt.savefig(filename)
    print(f'Wrote {filename}')
    return filename


async def main():
    executor = ThreadPoolExecutor(max_workers=len(repos))
    coroutines = [process_repo(repo, executor) for repo in repos]
    results = await asyncio.gather(*coroutines)
    executor.shutdown(wait=True)

if __name__ == "__main__":
    asyncio.run(main())
