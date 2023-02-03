#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Local imports,
from featexp import (
    get_trend_stats, 
    get_univariate_plots, 
    univariate_plotter
)

# #### Download data from here: https://www.kaggle.com/c/home-credit-default-risk/data

# In[2]:


# Functions for data preprocessing.
def get_nonull_dummy_data(
    application_train_raw, dummy_drop=["ORGANIZATION_TYPE"]
):
    # Idenifying float cols with less nulls and imputing with mean.
    nulls = pd.isnull(application_train_raw).sum()
    less_nulls = nulls[(nulls < 3075) & (nulls != 0)].index
    less_nulls_float = []
    for i in range(len(less_nulls)):
        if application_train_raw[less_nulls[i]].dtype != "O":
            less_nulls_float.append(less_nulls[i])

    application_train_raw[less_nulls_float] = application_train_raw[
        less_nulls_float
    ].fillna(application_train_raw[less_nulls_float].mean())

    # Idenifying float cols with high nulls and creating null_flag 
    # column and imputing with min-10.
    more_nulls = nulls[(nulls >= 3075)].index
    more_nulls_float = []
    for i in range(len(more_nulls)):
        if application_train_raw[more_nulls[i]].dtype != "O":
            more_nulls_float.append(more_nulls[i])

    application_train_raw[more_nulls_float] = application_train_raw[
        more_nulls_float
    ].fillna(application_train_raw[more_nulls_float].min() - 100)

    # Get dummies. Drop some columns for now.
    application_train_raw.drop(
        columns=dummy_drop, axis=1, inplace=True
    )  # Try using later.

    all_cols = application_train_raw.columns
    cat_cols = []
    for col in all_cols:
        if application_train_raw[col].dtype == "O":
            cat_cols.append(col)

    application_train_raw = pd.get_dummies(
        application_train_raw, columns=cat_cols, dummy_na=True
    )

    return application_train_raw


def import_and_create_train_test_data(test_size=0.33, random_state=42):
    application_raw = pd.read_csv("demo/data/application_train.csv")
    application = get_nonull_dummy_data(
        application_raw, dummy_drop=["ORGANIZATION_TYPE"]
    )

    X = application.drop(["TARGET"], axis=1)  # Contains ID.
    y = application["TARGET"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    train_users = X_train[["SK_ID_CURR"]]
    train_users["TARGET"] = y_train
    test_users = X_test[["SK_ID_CURR"]]
    test_users["TARGET"] = y_test
    train_users.reset_index(drop=True, inplace=True)
    test_users.reset_index(drop=True, inplace=True)

    return (X_train, X_test, y_train, y_test, train_users, test_users)


def import_and_create_TEST_data():
    application_raw = pd.read_csv("demo/data/application_test.csv")
    application = get_nonull_dummy_data(
        application_raw, dummy_drop=["ORGANIZATION_TYPE"]
    )

    X = application  # Contains ID.

    users = X[["SK_ID_CURR"]]
    users.reset_index(drop=True, inplace=True)

    return (X, users)


def get_imp_df(xgb_model):
    imp = pd.DataFrame(np.asarray(list(xgb_model.get_fscore().keys())))
    imp.columns = ["Feature"]
    imp["importance"] = np.asarray(list(xgb_model.get_fscore().values()))
    imp = imp.sort_values(by=["importance"], ascending=False)
    imp = imp.reset_index(drop=True)
    return imp

# In[3]:


X_train, X_test, y_train, y_test, train_users, test_users = import_and_create_train_test_data()
X_TEST, TEST_users = import_and_create_TEST_data()

drop=['CODE_GENDER_XNA', 'NAME_INCOME_TYPE_Maternity leave', 'NAME_FAMILY_STATUS_Unknown', 'SK_ID_CURR']
X_train = X_train.drop(drop, axis=1)
X_test = X_test.drop(drop, axis=1)

# In[4]:


data_train = X_train.reset_index(drop=True)
data_train['target'] = y_train.reset_index(drop=True)
data_test = X_test.reset_index(drop=True)
data_test['target'] = y_test.reset_index(drop=True)

# In[5]:


# plots univariate plots of first 10 columns in data_train
get_univariate_plots(data=data_train, target_col='target', features_list=data_train.columns[0:10], data_test=data_test)

# In[6]:


# Get grouped data, mean target and sample size of each group using univariate_plotter()
# With train and test data:
grouped_train, grouped_test = univariate_plotter(data=data_train, target_col='target', feature='AMT_INCOME_TOTAL',
                                                 data_test=data_test)

# With only train data
# grouped_train = univariate_plotter(data=data_train, target_col='target', feature='AMT_INCOME_TOTAL')

# In[7]:


grouped_train #Grouped data showing bin level stats

# ## Model trained using all features

# In[26]:


dtest = xgb.DMatrix(X_test, label=y_test, missing=np.nan)
dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.nan)

params = {'max_depth':8, 'learning_rate':0.1, 'silent':0, 'objective':'binary:logistic', 'min_child_weight':600,
            'eval_metric' : 'auc', 'nthread':16 } #col_sample_by_tree
xgb_model = xgb.train(params, dtrain, 400, evals=[(dtrain, 'train'), (dtest, 'test')], early_stopping_rounds=25) #, (dtest, 'test')


# In[27]:


dTEST = xgb.DMatrix(X_TEST[X_test.columns], missing=np.nan)
y_TEST_pred = xgb_model.predict(dTEST)
submission_all_feats = pd.DataFrame({'SK_ID_CURR' : TEST_users['SK_ID_CURR'], 'TARGET' : y_TEST_pred})
submission_all_feats.to_csv('submission_all_feats_1.csv', index=False)

# ## Calculating trend correlations and extracting feature importance from above model

# In[28]:


stats = get_trend_stats(data=data_train, target_col='target', data_test=data_test)
# 0 correlation is returned for constant valued features and hence get dropped based on low correlation criteria

# In[29]:


importance_df = get_imp_df(xgb_model) # get xgboost importances in dataframe
stats = pd.merge(stats, importance_df, how='left', on='Feature')
stats['importance'] = stats['importance'].fillna(0)

# ## Dropping features with trend corr < 0.95

# In[30]:


noisy = list(stats[stats['Trend_correlation']<0.95]['Feature'])
dtest = xgb.DMatrix(X_test.drop(noisy, axis=1), label=y_test, missing=np.nan)
dtrain = xgb.DMatrix(X_train.drop(noisy, axis=1), label=y_train, missing=np.nan)

params = {'max_depth':8, 'learning_rate':0.1, 'silent':0, 'objective':'binary:logistic', 'min_child_weight':600,
            'eval_metric' : 'auc', 'nthread':8 }
xgb_model = xgb.train(params, dtrain, 400, evals=[(dtrain, 'train'), (dtest, 'test')], early_stopping_rounds=25)


# In[31]:


dTEST = xgb.DMatrix(X_TEST[X_test.columns].drop(noisy, axis=1), missing=np.nan)
y_TEST_pred = xgb_model.predict(dTEST)
submission_95 = pd.DataFrame({'SK_ID_CURR' : TEST_users['SK_ID_CURR'], 'TARGET' : y_TEST_pred})
submission_95.to_csv('submission_95_1.csv', index=False)


# ## Dropping features with trend corr < 0.93

# In[32]:


noisy = list(stats[stats['Trend_correlation']<0.93]['Feature'])
dtest = xgb.DMatrix(X_test.drop(noisy, axis=1), label=y_test, missing=np.nan)
dtrain = xgb.DMatrix(X_train.drop(noisy, axis=1), label=y_train, missing=np.nan)

params = {'max_depth':8, 'learning_rate':0.1, 'silent':0, 'objective':'binary:logistic', 'min_child_weight':600,
            'eval_metric' : 'auc', 'nthread':16 }
xgb_model = xgb.train(params, dtrain, 400, evals=[(dtrain, 'train'), (dtest, 'test')], early_stopping_rounds=25)


# In[33]:


dTEST = xgb.DMatrix(X_TEST[X_test.columns].drop(noisy, axis=1), missing=np.nan)
y_TEST_pred = xgb_model.predict(dTEST)
submission_93 = pd.DataFrame({'SK_ID_CURR' : TEST_users['SK_ID_CURR'], 'TARGET' : y_TEST_pred})
submission_93.to_csv('submission_93_1.csv', index=False)

# ## Dropping features with trend corr < 0.90

# In[34]:


noisy = list(stats[stats['Trend_correlation']<0.90]['Feature'])
dtest = xgb.DMatrix(X_test.drop(noisy, axis=1), label=y_test, missing=np.nan)
dtrain = xgb.DMatrix(X_train.drop(noisy, axis=1), label=y_train, missing=np.nan)

params = {'max_depth':8, 'learning_rate':0.1, 'silent':0, 'objective':'binary:logistic', 'min_child_weight':600,
            'eval_metric' : 'auc', 'nthread':8 }
xgb_model = xgb.train(params, dtrain, 400, evals=[(dtrain, 'train'), (dtest, 'test')], early_stopping_rounds=25)

# In[35]:


dTEST = xgb.DMatrix(X_TEST[X_test.columns].drop(noisy, axis=1), missing=np.nan)
y_TEST_pred = xgb_model.predict(dTEST)
submission_90 = pd.DataFrame({'SK_ID_CURR' : TEST_users['SK_ID_CURR'], 'TARGET' : y_TEST_pred})
submission_90.to_csv('submission_90_1.csv', index=False)

# ## Dropping features with trend corr < 0.95 and feature importance < 40

# In[87]:


noisy = list(stats[(stats['Trend_correlation']<0.9) & (stats['importance']<10)]['Feature']) # 
dtest = xgb.DMatrix(X_test.drop(noisy, axis=1), label=y_test, missing=np.nan)
dtrain = xgb.DMatrix(X_train.drop(noisy, axis=1), label=y_train, missing=np.nan)

params = {'max_depth':8, 'learning_rate':0.1, 'silent':0, 'objective':'binary:logistic', 'min_child_weight':600,
            'eval_metric' : 'auc', 'nthread':16}
xgb_model = xgb.train(params, dtrain, 400, evals=[(dtrain, 'train'), (dtest, 'test')], early_stopping_rounds=25)
# [149]	train-auc:0.77722	test-auc:0.75991
# 0.74106 train-auc:0.78435	test-auc:0.76053 with <0.97

# In[88]:


dTEST = xgb.DMatrix(X_TEST[X_test.columns].drop(noisy, axis=1), missing=np.nan)
y_TEST_pred = xgb_model.predict(dTEST)
submission_95_40 = pd.DataFrame({'SK_ID_CURR' : TEST_users['SK_ID_CURR'], 'TARGET' : y_TEST_pred})
submission_95_40.to_csv('submission_95_40_1.csv', index=False)


# In[ ]:



