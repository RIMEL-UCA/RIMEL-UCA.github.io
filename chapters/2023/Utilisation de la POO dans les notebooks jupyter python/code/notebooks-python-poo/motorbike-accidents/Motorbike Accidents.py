import math

import numpy as np
import pandas as pd
import seaborn as sns
from fbprophet import Prophet

%matplotlib inline
from matplotlib import pyplot

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
DATASET_PATH = 'datasets/motorbike_ambulance_calls.csv'

pd.plotting.register_matplotlib_converters()
motorbike_data = (
    pd.read_csv(
        DATASET_PATH,
        parse_dates=['date'],
        dayfirst=False,
    )
    .set_index('index')
)
motorbike_data.info()
motorbike_data.head()
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

def convert_datetime_to_unix_timestamp(data):
    return (
        data
        .assign(datetime=lambda x: x['datetime'].astype(np.int64) // 10**9)
    )
analysis_data = (
    motorbike_data
    .pipe(add_datetime_feature)
    .pipe(convert_datetime_to_unix_timestamp)
)

analysis_data.head()
target_feature = 'cnt'
numerical_features = {
    'temp', 'atemp', 'hum', 'windspeed',
    'yr', 'mnth', 'hr', 'datetime'
}
numerical_and_target_features = numerical_features | set([target_feature])
categorical_features = {
    'season', 'holiday', 'weekday', 'weathersit', 'workingday'
}
leftout_features = (
    set(motorbike_data.columns) 
    - set([target_feature])
    - numerical_features
    - categorical_features
)
leftout_features
(
    analysis_data
    .reindex(columns=numerical_and_target_features)
    .describe()
)
(
    analysis_data
    .isnull()
    .any()
    .any()
)
(
    analysis_data
    .reindex(columns=numerical_and_target_features)
    .corr()
)
print('Max correlation for each feature:')
(
    analysis_data
    .reindex(columns=numerical_and_target_features)
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
sns.pairplot(
    (
        analysis_data
        .reindex(columns=numerical_and_target_features)
        .drop(columns='temp')
    )
);
sns.catplot(
    x='hr',
    y='cnt',
    kind='box',
    hue='workingday',
    data=analysis_data,
    aspect=2
);
sns.catplot(
    x='hr',
    y='cnt',
    kind='box',
    hue='workingday',
    row='season',
    data=analysis_data,
    aspect=2
);
sns.catplot(
    x='hr',
    y='cnt',
    kind='box',
    hue='workingday',
    row='weathersit',
    data=analysis_data,
    aspect=2
);
sns.catplot(
    x='weathersit',
    y='cnt',
    data=analysis_data,
    aspect=2
);
(
    analysis_data
    .groupby('weathersit')
    ['cnt']
    .sum()
    .plot(kind='bar', title='Number of accidents by weathersit')
);
sns.catplot(
    x='weekday',
    y='cnt',
    kind='box',
    hue='workingday',
    data=analysis_data,
    aspect=2
);
X, y = motorbike_data.drop(columns=target_feature), motorbike_data[target_feature]

# No time machine: use 'past' data for training, use 'future' data for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False
)

assert X_train.index.max() < X_test.index.min()
model_numerical_features = list(numerical_features - {'temp', 'datetime'})
print('Numerical features:', model_numerical_features)
model_categorical_features = list(categorical_features)
print('Categorical features:', model_categorical_features)
numerical_transformer = Pipeline(
    steps=[
        # In case serving data has missing values
        ('imputer', SimpleImputer(strategy='mean')),
        # Need to scale windspeed
        ('scaler', StandardScaler())
    ]
)
categorical_transformer = Pipeline(
    steps=[
        # In case serving data has missing values
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(categories='auto', 
                                 sparse=False, 
                                 handle_unknown='ignore'))
    ]
)
features_pipeline = Pipeline(steps=[
    (
        'features',
        ColumnTransformer(
            transformers=[
                (
                    'numerical',
                    numerical_transformer, 
                    model_numerical_features
                ),
                (
                    'categorical', 
                    categorical_transformer, 
                    model_categorical_features
                )
            ],
            remainder='drop'
        )
    )
])
def build_pipeline(model, use_grid_search=True, **grid_search_params):
    pipeline = Pipeline(steps=[
        ('features', features_pipeline),
        ('model', model)
    ])
    if use_grid_search:
        grid_search_params = {
            'cv': TimeSeriesSplit(n_splits=5),

            **grid_search_params
        }
        return GridSearchCV(pipeline, **grid_search_params)
    return pipeline
def evaluate_prediction(model_name, true_values, prediction):
    rmse = math.sqrt(
        mean_squared_error(true_values, prediction)
    )
    print(f'{model_name} RMSE: ', rmse)
    sns.relplot(
        x=model_name,
        y='true values',
        data=pd.DataFrame({
            model_name: prediction,
            'true values': true_values
        })
    );
def evaluate_model(model_name, model, **grid_search_params):
    pipeline = build_pipeline(
        model,
        use_grid_search='param_grid' in grid_search_params,
        **grid_search_params
    )
    pipeline.fit(X_train, y_train)
    evaluate_prediction(model_name, y_test, pipeline.predict(X_test))
    if hasattr(pipeline, 'best_params_'):
        print('Best params: ', pipeline.best_params_)
    return pipeline
from sklearn.linear_model import LinearRegression

linear_regression = evaluate_model('linear regression', LinearRegression())
print('R2 score: ', linear_regression.score(X_test, y_test))
from sklearn.tree import DecisionTreeRegressor

decision_tree = evaluate_model(
    'decision tree', 
    DecisionTreeRegressor(random_state=42),
    param_grid={
        # Already found the best params
        'model__max_depth': [20]
    }
)
from sklearn.ensemble import RandomForestRegressor

random_forest = evaluate_model(
    'random forest',
    RandomForestRegressor(random_state=42),
    param_grid={
        # Already found the best params
        'model__n_estimators': [40],
        'model__max_depth': [20]
    }
)
def _transform_data_for_prophet(X, y):
    df = (
        pd.concat(
            [
                (
                    X
                    .pipe(add_datetime_feature)
                    ['datetime']
                    .rename('ds')
                ),
                y.rename('y')
            ],
            axis='columns',
            sort=False
        )
    )
    holidays = (
        X
        .groupby('date', as_index=False)
        .first()
        .assign(holiday=lambda x: np.where(
            x['holiday'] == 1,
            'holiday',
            np.where(
                x['workingday'] == 0,
                'weekend',
                None
            )
        ))
        [lambda x: x['workingday'] == 0]
        [['date', 'holiday']]
        .rename(columns={
            'date': 'ds'
        })
    )
    return df, holidays

(
    X_train
    .pipe(_transform_data_for_prophet, y=y_train)
    
)
class ProphetRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(self, **prophet_args):
        self.prophet_args = prophet_args
        self.model = None
        self.last_prediction = None
    
    def fit(self, X, y=None):
        df, holidays = _transform_data_for_prophet(X, y)
        
        prophet_args = self.prophet_args.copy()
        if prophet_args.pop('use_holidays', None):
            prophet_args['holidays'] = holidays
            
        quarterly_seasonality = prophet_args.pop('quarterly_seasonality', None)
        montly_seasonality = prophet_args.pop('montly_seasonality', None)
        hourly_seasonality = prophet_args.pop('hourly_seasonality', None)
        
        self.model = Prophet(**prophet_args)
        
        if quarterly_seasonality:
            self.model.add_seasonality(
                name='quarterly',
                period=365.25 / 4,
                fourier_order=5
            )
        if montly_seasonality:
            self.model.add_seasonality(
                name='montly',
                period=30.5,
                fourier_order=5
            )
        if hourly_seasonality:
            self.model.add_seasonality(
                name='hourly',
                period=24,
                fourier_order=5
            )
            
        self.model.fit(df)
        
        return self
    
    def predict(self, X):
        if not self.model:
            raise RuntimeError('Neet to train first')
        future = (
            X
            .pipe(add_datetime_feature)
            [['datetime']]
            .rename(columns={
                'datetime': 'ds'
            })
        )
        self.last_prediction = self.model.predict(future)
        yhat = self.last_prediction['yhat'].copy()
        yhat.index = X.index
        return yhat
prophet1 = ProphetRegressor(
    yearly_seasonality=True,
    quarterly_seasonality=True,
    # Montly seasonality actually makes it worse.
#     montly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
    hourly_seasonality=True,
    use_holidays=True
)

prophet1.fit(X_train, y_train)
evaluate_prediction('prophet 1', y_test, prophet1.predict(X_test))
prophet1.model.plot(prophet1.last_prediction)
prophet1.model.plot_components(prophet1.last_prediction)
X_train_working = (
    X_train
    [X_train['workingday'] == 1]
)

y_train_working = (
    y_train
    [X_train['workingday'] == 1]
)

assert (X_train_working.index == y_train_working.index).all()

X_test_working = (
    X_test
    [X_test['workingday'] == 1]
)

y_test_working = (
    y_test
    [X_test['workingday'] == 1]
)

assert (X_test_working.index == y_test_working.index).all()


X_train_nonworking = (
    X_train
    [X_train['workingday'] == 0]
)

y_train_nonworking = (
    y_train
    [X_train['workingday'] == 0]
)

assert (X_train_nonworking.index == y_train_nonworking.index).all()

X_test_nonworking = (
    X_test
    [X_test['workingday'] == 0]
)
y_test_nonworking = (
    y_test
    [X_test['workingday'] == 0]
)

assert (X_test_nonworking.index == y_test_nonworking.index).all()
(
    pd.concat([X_train_working, y_train_working], axis=1)
    .pipe(add_datetime_feature)
    [['datetime', 'cnt']]
    [lambda x: (x['datetime'] >= '2011-06-01') & (x['datetime'] <= '2011-07-01')]
    .plot(
        x='datetime',
        y='cnt',
        figsize=(15, 4),
        xticks=pd.date_range(f'2011-06-01', '2011-07-01')
    )
)

(
    pd.concat([X_test_working, y_test_working], axis=1)
    .pipe(add_datetime_feature)
    [['datetime', 'cnt']]
    [lambda x: (x['datetime'] >= '2012-06-01') & (x['datetime'] <= '2012-07-01')]
    .plot(
        x='datetime',
        y='cnt',
        figsize=(15, 4),
        xticks=pd.date_range('2012-06-01', '2012-07-01')
    )
)
prophet_working_days = ProphetRegressor(
    yearly_seasonality=True,
    quarterly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
    hourly_seasonality=True,
)

prophet_working_days.fit(X_train_working, y_train_working)

prophet_working_days_prediction = (
    pd.concat(
        [
            X_test_working, 
            y_test_working,
            prophet_working_days.predict(X_test_working)
        ], 
        axis=1
    )
    .pipe(add_datetime_feature)
    [['datetime', 'cnt', 'yhat']]
)

evaluate_prediction(
    'prophet working days', 
    y_test_working, 
    prophet_working_days_prediction['yhat']
)
fig, ax = pyplot.subplots(figsize=(13, 10))

(
    prophet_working_days_prediction
    .plot(
        ax=ax,
        x='datetime',
        y='cnt',
        label='cnt',
        color='green',
        alpha=0.7
    )
)

(
    prophet_working_days_prediction
    .plot(
        ax=ax,
        x='datetime',
        y='yhat',
        label='yaht',
        color='blue',
        alpha=0.7
    )
)
class ProphetTransformer(ProphetRegressor, TransformerMixin):
    
    def __init__(self, **prophet_args):
        super().__init__(**prophet_args)
        
    def transform(self, X):
        return self.predict(X).values.reshape(-1, 1)
class LeveragingRandomForestRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(self, **model_args):
        self.model_args = model_args
        self.model = None
    
    def fit(self, X, y=None):
        
        self.model = RandomForestRegressor(**self.model_args)
        self.model.fit(X[:, 1:], X[:, 0])
        
        return self
    
    def predict(self, X):
        if not self.model:
            raise RuntimeError('Neet to train first')
        prediction = self.model.predict(X[:, 1:])
        return (X[:, 0] - prediction).reshape(-1, 1)
leveraging_transformer = Pipeline(steps=[
    (
        'features',
        ColumnTransformer(
            transformers=[
                (
                    'prophet',
                    ProphetTransformer(
                        yearly_seasonality=True,
                        quarterly_seasonality=True,
                        weekly_seasonality=True,
                        daily_seasonality=True,
                        hourly_seasonality=True,
                        use_holidays=True
                    ),
                    ['date', 'hr', 'holiday', 'workingday']
                ),
                (
                    'numerical',
                    numerical_transformer, 
                    model_numerical_features
                ),
                (
                    'categorical', 
                    categorical_transformer, 
                    model_categorical_features
                )
            ],
            remainder='drop'
        )
    )
])
leveraging_pipeline = Pipeline(
    steps=[
        ('leveraging_transformer', leveraging_transformer),
        (
            'leveraging_random_forest', 
            LeveragingRandomForestRegressor(
                n_estimators=40,
                max_depth=20
            )
        )
    ]
)
leveraging_pipeline.fit(X_train, y_train)
leveraging_pipeline_prediction = leveraging_pipeline.predict(X_test)
evaluate_prediction(
    'leveraging pipeline', 
    y_test, 
    leveraging_pipeline_prediction.reshape(1, -1)[0, :]
)

