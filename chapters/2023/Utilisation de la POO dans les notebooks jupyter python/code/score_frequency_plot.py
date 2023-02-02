import os

path = 'C:\\Users\\admin\\Desktop\\github-repository-scrapper\\github-repo-scraping\\github-repo-scraping'
path_no_poo = path + '\\notebooks-python-no-poo'
path_poo = path + '\\notebooks-python-poo'

# Collecting the scores

def get_scores(path):
    scores = []
    for directory in os.listdir(path):
        if os.path.isdir(path + '\\' + directory):
            file = 'score.txt'
            with open(path + '\\' + directory + '\\' + file, 'r') as f:
                content = f.read()
            # Grab score : Your code has been rated at 6.31/10
            # or : Your code has been rated at 10.00/10 (previous run: 6.31/10, +3.69)
            if 'previous run' in content:
                score = content.split('previous run: ')[0].split('Your code has been rated at ')[1].split('/')[0]
            else:
                if content == "" : 
                    score = -1
                else :
                    score = content.split('Your code has been rated at ')[1].split('/')[0]
            scores.append(float(score))
    return scores

scores_poo = get_scores(path_poo)
scores_no_poo = get_scores(path_no_poo)
                
frequency_poo = dict([(x, 0) for x in range(-1, 11)])
frequency_no_poo = dict([(x, 0) for x in range(-1, 11)])

for score in scores_poo:
    if score == -1:
        frequency_poo[-1] += 1
    else:
        frequency_poo[int(score)] += 1

for score in scores_no_poo:
    if score == -1:
        frequency_no_poo[-1] += 1
    else:
        frequency_no_poo[int(score)] += 1

# Plotting the scores
import matplotlib.pyplot as plt

x_poo = list(frequency_poo.keys())
y_poo = list(frequency_poo.values())
x_no_poo = list(frequency_no_poo.keys())
y_no_poo = list(frequency_no_poo.values())

plt.plot(x_poo, y_poo, label='poo')
plt.plot(x_no_poo, y_no_poo, label='no_poo')

plt.xlabel('Score')
plt.ylabel('Frequency')
plt.title('Score frequency distribution')

plt.legend()
plt.show()