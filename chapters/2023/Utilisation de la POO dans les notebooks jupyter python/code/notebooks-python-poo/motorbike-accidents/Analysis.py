import os
import itertools

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import poisson, norm, gamma
from scipy.special import factorial
import statsmodels.api as sm

from sklearn.metrics import mean_squared_error

%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns
DATASET_PATH = 'datasets'
FIGURES_PATH = 'figures'

pd.plotting.register_matplotlib_converters()
sns.set_style('whitegrid')
sns.palplot(sns.color_palette('muted'))
sns.set_palette('muted')
year_palette = {
    2011: 'C0',
    2012: 'C3'
}
season_order = ['winter', 'spring', 'summer', 'autumn']
motorbike_data = (
    pd.read_csv(
        os.path.join(DATASET_PATH, 'motorbike_ambulance_calls.csv'),
        parse_dates=['date'],
        dayfirst=False,
    )
    .set_index('index')
    
    .assign(
        yr=lambda x: np.where(
            x['yr'] == 0,
            2011,
            2012
        )
    )
    .assign(
        season=lambda x: (
            pd.Categorical(
                x['season'], 
                categories=season_order, 
                ordered=True
            )
        )
    )
)
motorbike_data.info()
motorbike_data.head()
if not os.path.exists(FIGURES_PATH):
    os.mkdir(FIGURES_PATH)
g = sns.relplot(
    x='mnth',
    y='cnt',
    hue='yr',
    legend='full',
    kind='line',
    marker='o',
    aspect=2,
    palette=year_palette,
    data=motorbike_data
)
g._legend.texts[0].set_text('Year')
g.set(
    xlabel='Month', 
    ylabel='Average number of calls', 
)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Number of calls by month and year')
g.fig.savefig(os.path.join(FIGURES_PATH, 'calls-by-month.png'))
data_by_year = (
    motorbike_data
    .groupby(['yr', 'mnth'], as_index=False)
    ['cnt'].mean()
)

data_11 = (
    data_by_year
    .loc[lambda x: x['yr'] == 2011, 'cnt']
    .reset_index(drop=True)
)

data_12 = (
    data_by_year
    .loc[lambda x: x['yr'] == 2012, 'cnt']
    .reset_index(drop=True)
)

def years_diff_loss(intercept):
    return mean_squared_error(data_11 + intercept, data_12)

optimum = minimize_scalar(years_diff_loss)
assert optimum.success
print(optimum)
g = sns.relplot(
    x='mnth',
    y='cnt',
    hue='yr',
    legend='full',
    kind='line',
    marker='o',
    aspect=2,
    palette=year_palette,
    data=(
        motorbike_data
        .assign(cnt=lambda x: np.where(
            x['yr'] == 2011,
            x['cnt'] + optimum.x,
            x['cnt']
        ))
    )
)
g._legend.texts[0].set_text('Year')
g.set(
    xlabel='Month', 
    ylabel='Average number of calls', 
)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Seasonality change of number of calls by month and year')
g.fig.savefig(os.path.join(FIGURES_PATH, 'seasonality-change-calls-by-month.png'))
(
    motorbike_data
    .groupby(['yr', 'season'])
    ['date']
    .agg(['min', 'max'])
)
g = sns.relplot(
    x='season',
    y='cnt',
    hue='yr',
    legend='full',
    kind='line',
    marker='o',
    aspect=2,
    palette=year_palette,
    data=motorbike_data
)
g._legend.texts[0].set_text('Year')
g.set(
    xlabel='Season', 
    ylabel='Average number of calls', 
)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Number of calls by season and year')
g.fig.savefig(os.path.join(FIGURES_PATH, 'calls-by-sesason.png'))
g = sns.catplot(
    x='season',
    y='cnt',
    hue='yr',
    legend='full',
    kind='bar',
    palette=year_palette,
    aspect=2,
    data=(
        motorbike_data
        .groupby(['yr', 'season'], as_index=False)
        ['cnt']
        .count()
    )
)
g._legend.set_title('Year')
g.set(
    xlabel='Season', 
    ylabel='Number of observations', 
)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Number of observations by season and year')
g.fig.savefig(os.path.join(FIGURES_PATH, 'observations-by-season.png'))
g = sns.relplot(
    x='hr',
    y='cnt',
    hue='yr',
    row='workingday',
    legend='full',
    kind='line',
    marker='o',
    aspect=2,
    palette=year_palette,
    data=motorbike_data
)
g._legend.texts[0].set_text('Year')
g.set(
    xlabel='Hour', 
    ylabel='Average number of calls',
    xticks=np.arange(0, 24)
)
for ax in g.axes.flatten():
    ax.tick_params(labelbottom=True)
plt.subplots_adjust(top=0.9, hspace=0.15)
g.fig.suptitle('Number of calls by hour and year')
g.fig.savefig(os.path.join(FIGURES_PATH, 'calls-by-hour.png'))
g = sns.catplot(
    x='hr',
    y='cnt',
    hue='yr',
    legend='full',
    kind='bar',
    palette=year_palette,
    aspect=2,
    data=(
        motorbike_data
        .groupby(['yr', 'hr'], as_index=False)
        ['cnt']
        .count()
    )
)
g._legend.set_title('Year')
g.set(
    xlabel='Hour', 
    ylabel='Number of observations', 
)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Number of observations by hour and year')
g.fig.savefig(os.path.join(FIGURES_PATH, 'observations-by-hour.png'))
g = sns.FacetGrid(
    motorbike_data, 
    col='yr',
    hue='yr',
    palette=year_palette,
    height=8
)
g.map(
    sns.pointplot, 
    'season', 
    'cnt', 
    order=season_order
)
g.axes[0][0].set(
    title='Year 2011',
    xlabel='Season',
    ylabel='Average number of calls'
)
g.axes[0][1].set(
    title='Year 2012',
    xlabel='Season'
)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Number of calls by season and year')
g.fig.savefig(os.path.join(FIGURES_PATH, 'calls-by-season-significance.png'))
g = sns.FacetGrid(
    motorbike_data, 
    col='yr',
    hue='yr',
    palette=year_palette,
    height=8
)
g.map(
    sns.pointplot, 
    'mnth', 
    'cnt',
    order=np.arange(1, 13)
)
g.axes[0][0].set(
    title='Year 2011',
    xlabel='Month',
    ylabel='Average number of calls'
)

g.axes[0][1].set(
    title='Year 2012',
    xlabel='Month'
)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Number of calls by month and year')
g.fig.savefig(os.path.join(FIGURES_PATH, 'calls-by-month-significance.png'))
g = sns.relplot(
    x='weathersit',
    y='cnt',
    hue='yr',
    legend='full',
    kind='line',
    marker='o',
    aspect=2,
    palette=year_palette,
    data=motorbike_data
)
g._legend.texts[0].set_text('Year')
g.set(
    xlabel='Weather situatuion', 
    ylabel='Average number of calls',
    xticks=[1, 2, 3, 4]
)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Number of calls by weather situation and year')
g.fig.savefig(os.path.join(FIGURES_PATH, 'calls-by-weathersit.png'))
g = sns.catplot(
    x='weathersit',
    y='cnt',
    hue='yr',
    legend='full',
    kind='bar',
    palette=year_palette,
    aspect=2,
    data=(
        motorbike_data
        .groupby(['yr', 'weathersit'], as_index=False)
        ['cnt']
        .count()
    )
)
g._legend.set_title('Year')
g.set(
    xlabel='Weather situation', 
    ylabel='Number of observations',
)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Number of observations by weather situation and year')
g.fig.savefig(os.path.join(FIGURES_PATH, 'observations-by-weathersit.png'))

(
    motorbike_data
    .groupby(['yr', 'weathersit'], as_index=False)
    ['cnt']
    .count()
)
g = sns.relplot(
    x='hr',
    y='cnt',
    hue='yr',
    row='weekday',
    col='holiday',
    legend='full',
    kind='line',
    marker='o',
    aspect=2,
    palette=year_palette,
    data=motorbike_data
)
g._legend.texts[0].set_text('Year')
g.set(
    xlabel='Hour', 
    ylabel='Average number of calls',
    xticks=np.arange(0, 24)
)
for ax in g.axes.flatten():
    ax.tick_params(labelbottom=True)
plt.subplots_adjust(top=0.95, hspace=0.15)
g.fig.suptitle('Number of calls by hour and year, split by weekday X holiday')
g.fig.savefig(os.path.join(FIGURES_PATH, 'calls-by-hour-by-weekday-by-holiday.png'))
g = sns.relplot(
    x='hr',
    y='cnt',
    hue='yr',
    row='season',
    col='weathersit',
    legend='full',
    kind='line',
    marker='o',
    aspect=2,
    palette=year_palette,
    data=motorbike_data
)
g._legend.texts[0].set_text('Year')
g.set(
    xlabel='Hour', 
    ylabel='Average number of calls',
    xticks=np.arange(0, 24)
)
for ax in g.axes.flatten():
    ax.tick_params(labelbottom=True)
plt.subplots_adjust(top=0.95, hspace=0.15)
g.fig.suptitle('Number of calls by hour and year, split by season X weathersit')
g.fig.savefig(os.path.join(FIGURES_PATH, 'calls-by-hour-by-season-by-weathersit.png'))
g = sns.catplot(
    x='weathersit',
    y='cnt',
    hue='yr',
    row='season',
    legend='full',
    kind='bar',
    palette=year_palette,
    aspect=2,
    data=(
        motorbike_data
        .groupby(['yr', 'weathersit', 'season'], as_index=False)
        ['cnt']
        .count()
    )
)
g._legend.set_title('Year')
g.set(
    xlabel='Weather situation', 
    ylabel='Number of observations',
)
for ax in g.axes.flatten():
    ax.tick_params(labelbottom=True)

plt.subplots_adjust(top=0.9, hspace=0.2)
g.fig.suptitle('Number of observations by weather situation, season and year')
g.fig.savefig(os.path.join(FIGURES_PATH, 'observations-by-weathersit-by-season.png'))

(
    motorbike_data
    .groupby(['yr', 'weathersit'], as_index=False)
    ['cnt']
    .count()
)
numerical_features = {
    'cnt', 'temp', 'atemp', 'hum', 'windspeed'
}
(
    motorbike_data
    .reindex(columns=numerical_features)
    .corr()
)
print('Max correlation for each feature:')
(
    motorbike_data
    .reindex(columns=numerical_features)
    .corr()
    .pipe(lambda x: x.subtract(np.diag([1.0] * len(x.columns))))
    .apply(
        lambda x: pd.Series({
            'feature': x.abs().idxmax(), 
            'corr': x[x.abs().idxmax()]
        }), 
        axis='columns'
    )
)
g = sns.pairplot(
    motorbike_data,
    kind='reg',
    diag_kind='kde',
    vars=numerical_features
)

plt.subplots_adjust(top=0.9)
g.fig.suptitle('Relations between numerical features')
g.fig.savefig(os.path.join(FIGURES_PATH, 'relations-numerical.png'))
plt.figure(figsize=(15, 5))
ax = sns.distplot(
    motorbike_data['cnt'],
)
ax.set(
    xlabel='Number of calls'
)
ax.figure.suptitle('Distribution of the number of calls')
ax.figure.savefig(os.path.join(FIGURES_PATH, 'cnt-distribution.png'))
plt.figure(figsize=(15, 5))
ax = sns.distplot(
    np.log(motorbike_data['cnt']),
)
ax.set(
    xlabel='Log number of calls'
)
ax.figure.suptitle('Distribution of log of the number of calls')
ax.figure.savefig(os.path.join(FIGURES_PATH, 'log-cnt-distribution.png'))
plt.figure(figsize=(15, 5))
ax = sns.distplot(
    np.sqrt(motorbike_data['cnt']),
)
ax.set(
    xlabel='Square root of number of calls'
)
ax.figure.suptitle('Distribution of square root of the number of calls')
ax.figure.savefig(os.path.join(FIGURES_PATH, 'sqrt-cnt-distribution.png'))
def neg_poisson_log_likelihood(data, lamb):
    likelihoods = poisson.pmf(data, lamb)
    likelihoods = likelihoods[likelihoods != 0]
    return -np.sum(np.log(likelihoods))

def mle_poisson(data):
    optimum = minimize(
        lambda params, data: neg_poisson_log_likelihood(data, params[0]),
        x0=np.ones(1),
        args=(data,),
        method='SLSQP'
    )
    print(optimum)
    assert optimum.success
    return optimum.x[0]

mle_poisson(np.random.poisson(lam=0.2, size=1000))
mle_poisson(np.random.poisson(lam=7.3, size=1000))
x_plot = np.arange(1000)
fig = plt.figure(figsize=(15, 5))
sns.distplot(motorbike_data['cnt'])
plt.plot(
    x_plot,
    poisson.pmf(x_plot, mle_poisson(motorbike_data['cnt']))
)

fig.suptitle('Attempt to fit Poisson distribution with MLE')
fig.savefig(os.path.join(FIGURES_PATH, 'fitted-poisson.png'))
def neg_norm_log_likelihood(data, loc, scale):
    likelihoods = norm.pdf(data, loc=loc, scale=scale)
    likelihoods = likelihoods[likelihoods != 0]
    return -np.sum(np.log(likelihoods))

def mle_norm(data):
    optimum = minimize(
        lambda params, data: neg_norm_log_likelihood(
            data, 
            loc=params[0], 
            scale=params[1]
        ),
        x0=np.array([0, 1]),
        args=(data,),
        method='SLSQP'
    )
    print(optimum)
    assert optimum.success
    return optimum.x

mle_norm(np.random.normal(loc=3, scale=2, size=1000))
mle_norm(np.random.normal(loc=-2, scale=0.3, size=1000))
mle_norm(motorbike_data['cnt'])
fig = plt.figure(figsize=(15, 5))
ax = sns.distplot(
    motorbike_data['cnt'], kde=False,
    fit=norm
)
ax.set(
    xlabel='Number of calls'
)
ax.figure.suptitle('Fitted normal distribution')
ax.figure.savefig(os.path.join(FIGURES_PATH, 'fitted-normal.png'))
fig = plt.figure(figsize=(15, 5))
ax = sns.distplot(
    motorbike_data['cnt'], kde=False,
    fit=gamma
)
ax.set(
    xlabel='Number of calls'
)
ax.figure.suptitle('Fitted gamma distribution')
ax.figure.savefig(os.path.join(FIGURES_PATH, 'fitted-gamma.png'))
(
    motorbike_data
    ['date']
    .apply(['min', 'max'])
)
def reindex_by_hour(data):
    return (
        data
        .reindex(pd.MultiIndex.from_product(
            [
                pd.date_range('2011-01-01', '2012-12-31'),
                np.arange(24)
            ],
            names=['date', 'hr']
        ))
    )
observations_by_hour = (
    motorbike_data
    .set_index(['date', 'hr'])
    .pipe(reindex_by_hour)
)

missing_observations = (
    observations_by_hour
    [lambda x: x.isnull().any(axis='columns')]
    .index.to_frame()
)

missing_observations
(
    missing_observations
    .groupby(lambda index: index[0].year)
    .count()
)
missing_periods = (
    observations_by_hour
    .reset_index()
    .fillna({'cnt': -1})
    .assign(group_mask=lambda x: (x['cnt'] != x['cnt'].shift()).cumsum())
    .groupby(['cnt', 'group_mask'], as_index=False, sort=False)
    .count()
    .loc[lambda x: x['cnt'] == -1, 'hr']
)

fig = plt.figure(figsize=(15, 5))
sns.distplot(
    missing_periods, 
    kde=False
)
fig.suptitle('Distribution of periods of missing data')
fig.savefig(os.path.join(FIGURES_PATH, 'missing-periods.png'))

missing_periods.value_counts().sort_index()
fill_limit = None

cleaned_data = (
    motorbike_data
    .set_index(['date', 'hr'])
    .pipe(reindex_by_hour)
    .assign(
        my_yr=lambda x: x.index.get_level_values(0).year,
        my_mnth=lambda x: x.index.get_level_values(0).month
    )
)

assert len(cleaned_data) == len(motorbike_data) + len(missing_observations)
assert (
    cleaned_data
    [lambda x: pd.notnull(x['yr']) & pd.notnull(x['mnth'])]
    .pipe(lambda x: (x['yr'] == x['my_yr']) & (x['mnth'] == x['my_mnth']))
    .all()
)

cleaned_data = (
    cleaned_data
    .assign(
        yr=lambda x: x['my_yr'],
        mnth=lambda x: x['my_mnth']
    )
    .drop(columns=['my_yr', 'my_mnth'])
)

# Fill from group
columns_to_fill_from_group = [
    'season', 'holiday', 'weekday', 'workingday', 'weathersit'
]

filled_columns = (
    cleaned_data
    .groupby('date', as_index=False)
    [columns_to_fill_from_group]
    .transform(lambda x: (
        x.reset_index(drop=True)
        # limit for fillna() is not implemented yet
        .fillna(method='ffill')
        .fillna(method='bfill')
    ))
)

not_null_filled_columns_mask = (
    cleaned_data
    [columns_to_fill_from_group]
    .notnull()
    .all(axis='columns')
)
    
assert (
    filled_columns
    .loc[not_null_filled_columns_mask, columns_to_fill_from_group]
    == 
    cleaned_data
    .loc[not_null_filled_columns_mask, columns_to_fill_from_group]
).all().all()

cleaned_data[columns_to_fill_from_group] = filled_columns[columns_to_fill_from_group]

# Interpolate
columns_to_interpolate = ['temp', 'atemp', 'hum', 'windspeed', 'cnt']

filled_columns = (
    cleaned_data
    [columns_to_interpolate]
    .apply(
        lambda x: (
            x.interpolate(method='linear', limit=fill_limit)
        ),
        axis='index'
    )
)

not_null_filled_columns_mask = (
    cleaned_data
    [columns_to_interpolate]
    .notnull()
    .all(axis='columns')
)

assert (
    filled_columns
    .loc[not_null_filled_columns_mask, columns_to_interpolate]
    == 
    cleaned_data
    .loc[not_null_filled_columns_mask, columns_to_interpolate]
).all().all()

cleaned_data[columns_to_interpolate] = filled_columns[columns_to_interpolate]

cleaned_data = (
    cleaned_data
    .assign(was_missing=lambda x: (
        pd.Series(True, index=missing_observations.index)
    ))
    .fillna({'was_missing': False})
    .reset_index()
)
cleaned_data.to_csv(
    os.path.join(DATASET_PATH, 'cleanted_motorbike_ambulance_calls.csv'),
    index=False
)

cleaned_data
observations_with_missing_data = (
    cleaned_data
    .set_index(['date', 'hr'])
    .pipe(reindex_by_hour)
    [lambda x: x.isnull().any(axis='columns')]
)
print(len(observations_with_missing_data))
observations_with_missing_data
def add_datetime_feature(data):
    return (
        data
        .assign(datetime=lambda x: (
            x.apply(
                lambda y: (
                    y['date'].replace(hour=y['hr'])
                ), 
                axis='columns'
            )
        ))
    )
(
    missing_observations
    .reset_index(drop=True)
    .groupby('date')
    .count()
    .nlargest(10, columns='hr')
)
def plot_missing_data(ax, column, date_lower, date_upper):
    data = (
        cleaned_data
        .pipe(add_datetime_feature)
        [lambda x: (x['datetime'] >= date_lower) & (x['datetime'] <= date_upper)]
    )

    sns.lineplot(
        ax=ax,
        x='datetime',
        y=column,
        color=sns.color_palette()[0],
        data=(
            data
            [lambda x: ~x['was_missing']]
        )
    )

    sns.scatterplot(
        ax=ax,
        x='datetime',
        y=column,
        color=sns.color_palette()[3],
        data=(
            data
            [lambda x: x['was_missing']]
        )
    )

missing_data_periods_of_interest = [
    ('2012-10-20', '2012-11-10'),
    ('2011-01-01', '2011-01-30')
]

for index, (date_lower, date_upper) in enumerate(missing_data_periods_of_interest):
    fig, axes = plt.subplots(
        len(columns_to_interpolate),
        figsize=(15, 25),
        sharex=True
    )
    plt.subplots_adjust(
        top=0.95,
        hspace=0.3
    )
    
    for ax, column in zip(axes, columns_to_interpolate):
        plot_missing_data(ax, column, date_lower, date_upper)
        ax.tick_params(labelbottom=True)
        
    fig.suptitle(f'Interpolated missing values, in red, from {date_lower} till {date_upper}')
    fig.savefig(os.path.join(FIGURES_PATH, f'interpolated-missing-values-{index}.png'))

