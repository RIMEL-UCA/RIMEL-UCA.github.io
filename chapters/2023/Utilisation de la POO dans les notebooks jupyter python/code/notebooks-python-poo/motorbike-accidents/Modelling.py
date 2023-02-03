import os
import math

import numpy as np
import pandas as pd
import seaborn as sns

%matplotlib inline
from matplotlib import pyplot as plt

from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
)
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
DATASET_PATH = 'datasets'
FIGURES_PATH = 'figures'

pd.plotting.register_matplotlib_converters()
sns.set_style('whitegrid')
sns.set_palette('muted')
sns.palplot(sns.color_palette())

season_order = ['winter', 'spring', 'summer', 'autumn']
motorbike_data = (
    pd.read_csv(
        os.path.join(DATASET_PATH, 'cleanted_motorbike_ambulance_calls.csv'),
        parse_dates=['date'],
        dayfirst=False,
    )
    .assign(
        yr=lambda x: np.where(
            x['yr'] == 2011,
            0,
            1
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
    .drop(columns='was_missing')
)
motorbike_data.info()
motorbike_data.head()
X, y = motorbike_data.drop(columns='cnt'), motorbike_data['cnt']

# No time machine: use 'past' data for training, use 'future' data for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False
)

assert X_train.index.max() < X_test.index.min()

print('Train: ', X_train[['date', 'hr']].nlargest(1, columns=['date', 'hr']))
print('Test: ', X_test[['date', 'hr']].nsmallest(1, columns=['date', 'hr']))
class FeatureMeanStdTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, feature, lags=None):
        self.feature = feature
        self.feature_mean_and_std = None
        self.lags = lags
        
    def _get_lags(self):
        return sorted(self.lags) if self.lags else [0]
        
    def fit(self, X, y=None):
        self.feature_mean_and_std = (
            X[['hr', self.feature]]
            .groupby('hr')
            [self.feature]
            .agg(['mean', 'std'])
            .rename(columns={
                'mean': f'{self.feature}_mean', 
                'std': f'{self.feature}_std'
            })
        )    
        return self
    
    def transform(self, X):
        if self.feature_mean_and_std is None:
            raise RuntimeError('Need to fit() first!')
        
        data_with_feature = (
            X
            [['date', 'hr']]
            .merge(
                self.feature_mean_and_std,
                how='left',
                left_on='hr',
                right_index=True
            )
            .sort_values(['date', 'hr'])
        )
        
        for lag in self._get_lags():
            if lag == 0:
                continue
            data_with_feature = (
                data_with_feature
                .assign(**{
                    f'{self.feature}_mean_{lag}h_lag': lambda x: (
                        x[f'{self.feature}_mean']
                        .shift(
                            lag, 
                            fill_value=x.iloc[:lag][f'{self.feature}_mean'].mean()
                        )
                    ),
                    f'{self.feature}_std_{lag}h_lag': lambda x: (
                        x[f'{self.feature}_std']
                        .shift(
                            lag, 
                            fill_value=x.iloc[:lag][f'{self.feature}_std'].mean()
                        )
                    )
                })
            )
            
        if 0 not in self._get_lags():
            data_with_feature = (
                data_with_feature
                .drop(columns=[f'{self.feature}_mean', f'{self.feature}_std'])
            )
        
        return (
            # We sorted, so we have to restore the original ordering
            X
            [['date', 'hr']]
            .merge(
                data_with_feature,
                how='left',
                on=['date', 'hr']
            )
            .drop(columns=['date', 'hr'])
        )
    
    def get_feature_names(self, in_names=None):
        feature_names = []
        for lag in self._get_lags():
            if lag == 0:
                feature_names.extend([
                    f'{self.feature}_mean', 
                    f'{self.feature}_std'
                ])
            else:
                feature_names.extend([
                    f'{self.feature}_mean_{lag}h_lag', 
                    f'{self.feature}_std_{lag}h_lag'
                ])
        return feature_names
    
(
    FeatureMeanStdTransformer('hum', lags=[1, 2, 3, 0])
    .fit_transform(X_train, y_train)
    .head()
)
class CntMeanStdTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, lags=None):
        self.lags = lags
        self.feature_transformer = None
        
    def fit(self, X, y=None):
        if y is None:
            raise RuntimeError('Target variable is required for fitting!')
            
        self.feature_transformer = FeatureMeanStdTransformer('cnt', self.lags)    
        
        data = (
            pd.concat(
                [X['hr'], y],
                axis='columns',
                sort=False
            )
        )
        self.feature_transformer.fit(data)
        return self
    
    def transform(self, X):
        if self.feature_transformer is None:
            raise RuntimeError('Need to fit() first!')
        return self.feature_transformer.transform(X)
    
    def get_feature_names(self, in_names=None):
        if self.feature_transformer is None:
            raise RuntimeError('Need to fit() first!')
        return self.feature_transformer.get_feature_names(in_names)
    
(
    CntMeanStdTransformer(lags=[1, 2, 3, 0])
    .fit_transform(X_train, y_train)
    .head()
)
class FeatureLagTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, feature, lags):
        self.feature = feature
        self.lags = lags
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        sorted_data = (
            X
            [['date', 'hr', self.feature]]
            .sort_values(['date', 'hr'])
        )
        
        for lag in self.lags:
            sorted_data = (
                sorted_data
                .assign(**{
                    f'{self.feature}_{lag}h_lag': lambda x: (
                        x[f'{self.feature}']
                        .shift(lag)
                    )
                })
            )
            
        return (
            # We sorted, so we have to restore the original ordering
            X
            [['date', 'hr']]
            .merge(
                sorted_data,
                how='left',
                on=['date', 'hr']
            )
            .drop(columns=['date', 'hr', self.feature])
        )
    
    def get_feature_names(self, in_names=None):
        feature_names = []
        for lag in self.lags:
            feature_names.append(f'{self.feature}_{lag}h_lag')
        return feature_names

(
    FeatureLagTransformer('hum', lags=[1, 2, 3])
    .fit_transform(X_train, y_train)
    .head()
)
class TrafficStateTransformer(BaseEstimator, TransformerMixin):
    """
    For working days:
        - [7, 9] U [16, 19] - rush hour
        - [10, 15] U [20, 21] - usual traffic
        - [0, 6] U [22, 23] - low traffic
        
    For non-working days:
        - [11, 17] - rush hour
        - [9, 10] U [18, 20] - usual traffic
        - [0, 8] U [21, 23] - low traffic
    """
    
    traffic_by_hour = (
        pd.DataFrame(
            {
                True: (
                    ['low_traffic'] * 7 
                    + ['rush_hour'] * 3 
                    + ['usual_traffic'] * 6 
                    + ['rush_hour'] * 4
                    + ['usual_traffic'] * 2
                    + ['low_traffic'] * 2
                ),
                False: (
                    ['low_traffic'] * 9
                    + ['usual_traffic'] * 2
                    + ['rush_hour'] * 7
                    + ['usual_traffic'] * 3
                    + ['low_traffic'] * 3
                )
            },
            index=pd.Int64Index(np.arange(24), name='hr'),
            columns=pd.Index([True, False], name='workingday')
        )
    )
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return (
            X
            [['hr', 'workingday']]
            .assign(
                traffic_state=lambda x: x.apply(
                    lambda y: (
                        TrafficStateTransformer.traffic_by_hour
                        .loc[y['hr'], y['workingday'] == 1]
                    ),
                    axis='columns'
                )
            )
            .drop(columns=['hr', 'workingday'])
        )
    
    def get_feature_names(self, in_names=None):
        return ['traffic_state']

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(TrafficStateTransformer.__doc__)
    display(TrafficStateTransformer.traffic_by_hour.T)
    display(
        X_train
        .set_index(['date', 'hr'], drop=False)
        .groupby('workingday', as_index=False)
        .head(24)
        .pipe(TrafficStateTransformer().transform)
        .transpose()
    )
class CntYearlyChange(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.mean_change = None
        self.std_change = None
        
    def fit(self, X, y=None):
        if y is None:
            raise RuntimeError('Target feature is required for fitting')
            
        self.mean_change, self.std_change = (
            pd.concat(
                [X[['yr', 'mnth']], y],
                axis='columns',
                sort=False
            )
            .groupby(['yr', 'mnth'])
            ['cnt']
            .agg(['mean', 'std'])
            .unstack('yr')
            .swaplevel(axis='columns')
            .pipe(lambda x: x[1] - x[0])
            .mean()
        )
        return self
    
    def transform(self, X):
        if self.mean_change is None or self.std_change is None:
            raise RuntimeError('Need to fit() first!')
            
        return (
            X[['yr']]
            .assign(
                cnt_yearly_mean_change=lambda x: self.mean_change * x['yr'],
                cnt_yearly_std_change=lambda x: self.std_change * x['yr']
            )
            .drop(columns='yr')
        )
    
    def get_feature_names(self, in_names=None):
        return ['cnt_yearly_mean_change', 'cnt_yearly_std_change']
    
(
    CntYearlyChange()
    .fit_transform(X_train, y_train)
)
class NamedFeaturesPipeline(Pipeline):
    
    def get_feature_names(self, in_names=None):
        feature_names = in_names
        for step_name, step in self.steps:
            try:
                feature_names = step.get_feature_names(feature_names)
            except AttributeError as exc:
                print(f'Beware: {step_name} does not have get_feature_names(): {exc}')
        return feature_names
    
class NamedFeaturesFeatureUnion(FeatureUnion):
    
    def get_feature_names(self, in_names=None):
        feature_names = []
        for step_name, step in self.transformer_list:
            feature_names.extend(step.get_feature_names(in_names))
        return feature_names

class NamedFeaturesColumnTransformer(ColumnTransformer):
    
    def get_feature_names(self, in_names=None):
        passthrough_features = []
        if in_names is not None:
            passthrough_features = list(in_names)
        else:
            passthrough_features = []
            
        feature_names = []
        for step_name, step, step_features in self.transformers_:
            if step_name == 'remainder':
                continue
            passthrough_features = (
                [x for x in passthrough_features if x not in step_features]
            )
            print(f'At {step_name} with features {step_features}')
            feature_names.extend(step.get_feature_names(step_features))
            
        if self.remainder == 'passthrough':
            feature_names.extend(passthrough_features)
        return feature_names

class NamedFeaturesNotChangedMixin:
    
    def get_feature_names(self, in_names=None):
        return in_names
    
class NamedFeaturesSimpleImputer(NamedFeaturesNotChangedMixin, SimpleImputer):
    pass

class NamedFeaturesStandardScaler(NamedFeaturesNotChangedMixin, StandardScaler):
    pass
def build_basic_features_pipeline():
    return NamedFeaturesColumnTransformer(
            [
                (
                    'numerical_features',
                    NamedFeaturesPipeline([
                        ('imputer', NamedFeaturesSimpleImputer(strategy='mean')),
                        ('scaler', NamedFeaturesStandardScaler())
                    ]),
                    ['temp', 'hum', 'windspeed']
                ),
                (
                    'categorical_features',
                    NamedFeaturesPipeline([
                        ('imputer', NamedFeaturesSimpleImputer(strategy='most_frequent')),
                        (
                            'onehot', 
                            OneHotEncoder(
                                categories='auto', 
                                sparse=False, 
                                handle_unknown='ignore'
                            )
                        )
                    ]),
                    ['hr', 'yr', 'mnth', 'season', 'weekday', 'weathersit']
                ),
                (
                    'unmodified_features',
                    NamedFeaturesSimpleImputer(strategy='most_frequent'),
                    ['holiday', 'workingday']
                )
            ],
            remainder='drop'
        )

def build_custom_features_pipeline():
    return NamedFeaturesFeatureUnion([
        (
            'numerical_features', 
            NamedFeaturesPipeline([
                (
                    'custom_numerical_features',
                    NamedFeaturesFeatureUnion([
                        (
                            'cnt_mean_std', 
                            CntMeanStdTransformer(lags=[0, 1, 2, 3, 6, 12])
                        ),
                        (
                            'hum_mean_std', 
                            FeatureMeanStdTransformer('hum', lags=[0, 1, 2, 3, 6, 12])
                        ),
                        (
                            'temp_mean_std', 
                            FeatureMeanStdTransformer('temp', lags=[0, 1, 2, 3, 6, 12])
                        ),

                        (
                            'hum_lag', 
                            FeatureLagTransformer('hum', lags=[1, 2, 3])
                        ),
                        (
                            'temp_lag', 
                            FeatureLagTransformer('temp', lags=[1, 2, 3])
                        ),
                        (
                            'windspeed_lag', 
                            FeatureLagTransformer('windspeed', lags=[1, 2, 3])
                        ),
#                         (
#                             'cnt_yearly_change',
#                             CntYearlyChange()
#                         )
                    ])
                ),
                ('imputer', NamedFeaturesSimpleImputer(strategy='mean')),
                ('scaler', NamedFeaturesStandardScaler())
            ])
        ),
        (
            'categorical_features', 
            NamedFeaturesPipeline([
                (
                    'custom_categorical_features',
                    NamedFeaturesFeatureUnion([
                        (
                            'weathersit_lag', 
                            FeatureLagTransformer('weathersit', lags=[1, 3])),
                        (
                            'traffic_state', 
                            TrafficStateTransformer()
                        )
                    ])
                ),
                ('imputer', NamedFeaturesSimpleImputer(strategy='most_frequent')),
                (
                    'onehot', 
                    OneHotEncoder(
                        categories='auto', 
                        sparse=False, 
                        handle_unknown='ignore'
                    )
                )
            ])
        )
    ])
def build_features_pipeline():
    return NamedFeaturesFeatureUnion([
        (
            'basic_features',
            build_basic_features_pipeline()
        ),
        (
            'custom_features',
            build_custom_features_pipeline()
        )
    ])
p = build_features_pipeline().fit(X_train, y_train)
feature_names = p.get_feature_names(X_train.columns)
assert p.transform(X_train).shape[1] == len(feature_names)
t_X_train = pd.DataFrame(p.transform(X_train), columns=feature_names)
print('Number of features: ', len(feature_names))
feature_names
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(X_train[:24])
    display(
        t_X_train
        [:24]
    )
    display(X_train[48:72])
    display(
        t_X_train
        [48:72]
    )
def build_pipeline(model, use_grid_search=True, **grid_search_params):
    pipeline = Pipeline([
        ('features', build_features_pipeline()),
        ('model', model)
    ])
    if use_grid_search:
        grid_search_params = {
            'cv': TimeSeriesSplit(n_splits=5),

            **grid_search_params
        }
        return GridSearchCV(pipeline, **grid_search_params)
    return pipeline
def evaluate_prediction(model_name, y_true, y_pred):
    rmse = math.sqrt(
        mean_squared_error(y_true, y_pred)
    )
    print(f'{model_name} RMSE: ', rmse)
    print(f'{model_name} R2 score: ', r2_score(y_true, y_pred))
    sns.relplot(
        x=model_name,
        y='true values',
        data=pd.DataFrame({
            'true values': y_true,
            model_name: y_pred
        })
    )
def evaluate_model(model_name, model, **grid_search_params):
    use_grid_search = 'param_grid' in grid_search_params
    pipeline = build_pipeline(
        model,
        use_grid_search=use_grid_search,
        **grid_search_params
    )
    pipeline.fit(X_train, y_train)
    print('On Train dataset:')
    evaluate_prediction(model_name, y_train, pipeline.predict(X_train))
    print('On Test dataset:')
    evaluate_prediction(model_name, y_test, pipeline.predict(X_test))
    if use_grid_search:
        print('Best params: ', pipeline.best_params_)
        estimator = pipeline.best_estimator_
    else:
        estimator = pipeline
    
    if hasattr(estimator['model'], 'feature_importances_'):
        feature_importance = (
            pd.DataFrame({
                'feature': feature_names,
                'importance': estimator['model'].feature_importances_
            })
            .sort_values(by='importance', ascending=False)
            .nlargest(20, columns='importance')
        )

        g = sns.catplot(
            x='importance',
            y='feature',
            kind='bar',
            aspect=2,
            data=feature_importance
        )
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(f'Feature importance of {model_name}')
        g.fig.savefig(
            os.path.join(
                FIGURES_PATH, 
                f'{model_name.replace(" ", "-")}-feature-importance.png'
            )
        )

    return estimator
random_forest = evaluate_model(
    'random forest',
    RandomForestRegressor(
        random_state=42, 
        n_estimators=1000, 
        max_depth=20, 
        max_features=0.5
    ),
#     param_grid={
#         'model__n_estimators': [750, 1000, 1250],
#         'model__max_depth': [20, 30],
#         'model__max_features': [0.5]
#     }
)
gbr = evaluate_model(
    'gradient boosting',
    GradientBoostingRegressor(
        random_state=42, 
        n_estimators=800,
        learning_rate=0.05,
        subsample=0.5
    ),
#     param_grid={
#         'model__n_estimators': [450, 500, 550, 600, 650, 700, 750, 800],
#         'model__learning_rate': [0.01, 0.05, 0.1],
#         'model__subsample': [0.5, 1]
#     }
)
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
model_predictions = (
    pd.DataFrame({
        'y_true': y_test,
        'y_pred': gbr.predict(X_test),
        'date': X_test['date'],
        'hr': X_test['hr']
    })
    .pipe(add_datetime_feature)
    .drop(columns=['date', 'hr'])
    .melt(id_vars=['datetime'], value_vars=['y_true', 'y_pred'])
    
    [lambda x: x['datetime'] < '2012-05-30']
)

model_predictions
g = sns.relplot(
    x='datetime',
    y='value',
    hue='variable',
    kind='line',
    palette={
        'y_true': 'C0',
        'y_pred': 'C3'
    },
    aspect=2,
    data=model_predictions
)

g.fig.suptitle('True and predicted values for the number of ambulance calls')
plt.subplots_adjust(top=0.95)
g.set(
    ylabel='Number of ambulance calls',
    xlabel='Date and hour'
)
g._legend.texts[0].set_text('Number of calls')
g._legend.texts[1].set_text('True value')
g._legend.texts[2].set_text('Predicted value')

g.fig.savefig(os.path.join(FIGURES_PATH, 'true-predicted-cnt.png'))

