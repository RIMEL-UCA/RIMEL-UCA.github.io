#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.session import Environment, GsSession
# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None)

# # Factor Models
# 
# The GS Quant `FactorRiskModel` class gives users the power to upload their own risk models to Marquee for seamless integration with the Marquee Portfolio Analytics and Plot Tool Pro suite. After uploading a custom `FactorRiskModel`, users can access their factor model data programmatically using GS Quant, visualize their factor risk model data with Plot Tool Pro, or run historical factor attribution analysis on equity portfolios through the lens of their uploaded factor risk model with GS Quant's `Portfolio` class.

# ## Create a Factor Model
# 
# Input fields to create the initial Factor Risk Model object
# 
# | Attribute       |Can be Modified    |Description
# |-----------------|-------------------|-------------
# | id              | No                |Model id|
# | name            | Yes               |Name of model|
# | description     | Yes               |Longer description of model|
# | term            | Yes               |Term or horizon of model. One of: Long, Medium, Short|
# | coverage        | Yes               |Geographical coverage of assets within model universe. One of: Global, Region, Region Excluding Countries, Country|
# | vendor          | Yes               |Who creates the model|
# | version         | Yes               |Version of model|
# | identifier      | No                |Identifier used to upload the model's asset universe. One of: sedol, cusip, bcid, gsid|
# | entitlements    | Yes               |Who can manage, edit, and view the risk model|
# 

# In[ ]:


from gs_quant.models.risk_model import FactorRiskModel, RiskModelCalendar, Term, CoverageType, UniverseIdentifier
from gs_quant.entities.entitlements import Group, Entitlements, User, EntitlementBlock

model_id = 'MY_MODEL'
model_name = 'My Risk Model'
description = 'My Custom Factor Risk Model'
term = Term.Medium
coverage = CoverageType.Country
universe_identifier = UniverseIdentifier.sedol
vendor = 'Goldman Sachs'

users = User.get_many(emails=["first_user@email.com", "second_user@email.com"])
groups = Group.get_many(group_ids=["my_marquee_group1","my_marquee_group_2"])

entitlements = Entitlements(
    admin=EntitlementBlock(users=users),
    edit=EntitlementBlock(users=users),
    upload=EntitlementBlock(users=users),
    query=EntitlementBlock(groups=groups, users=users),
    execute=EntitlementBlock(groups=groups, users=users),
    view=EntitlementBlock(groups=groups, users=users)
)

# Notes on entitlements for risk models:
# 
# | Entitlement     |Description
# |-----------------|-------------------
# | admin           | Can edit this model's entitlements
# | edit            | Can edit this model's metadata
# | upload          | Can upload data to this model (raw data)
# | query           | Can query this model's uploaded data (raw data)
# | execute         | Can run risk reports with this model (derived data)
# | view            | Can view risk reports run with this model
# 
# 

# In[ ]:


# create model with inputs

model = FactorRiskModel(
    id_=model_id,
    name=model_name,
    description=description,
    coverage=coverage,
    term=term,
    universe_identifier=universe_identifier,
    vendor=vendor,
    entitlements=entitlements,
    version=1,
)

model.save()

# ## Upload a Calendar To Your Model
# The calendar associated with the Factor Risk Model contains the dates which the risk model should have posted data on to be considered "complete." The calendar can go further back as well as forward in time than the data that is currently posted for the calendar, but there cannot be any gaps in the data posted to the risk model according to the calendar.

# In[ ]:


calendar = RiskModelCalendar([
    '2021-01-29', '2021-01-28', '2021-01-27', '2021-01-26', '2021-01-25', '2021-01-22', '2021-01-21',
    '2021-01-20', '2021-01-19', '2021-01-18', '2021-01-15', '2021-01-14', '2021-01-13', '2021-01-12',
    '2021-01-11', '2021-01-08', '2021-01-07', '2021-01-06', '2021-01-05', '2021-01-04', '2021-01-01'
])

model.upload_calendar(calendar)

# ## Upload Data To Your Model
# 
# Once the calendar is posted for a model, we can start uploading data to it. We can supply data multiple ways:
# 
# 1. Upload total data one day at a time
# 2. Upload partial data one day at a time
# 
# For a complete day of data, we need three things, defined in `RiskModelData`
# 1. Factor Data
#    - factorId: Can be any string, but needs to map consistently to the same factor across every date
#    - factorName: Can be any string, will be the display name of the factor, should be consistent across every date
#    - factorCategoryId: Id of the category that the factor belongs to
#    - factorCategory: Name of the category that the factor belongs to, will be the display name of the category (Style, Industry, Market, Currency, ect.)
#    - factorReturn: Daily return of the factor in percent units
# 2. Asset Data
#    - universe: Array of assets in the universe
#    - factorExposure: Array of dictionaries that map factorId to the factor exposure of that asset, corresponds to ordering of asset universe
#    - specificRisk: Array of annualized specific risk in percent units, corresponds to ordering of asset universe (null values not allowed)
#    - totalRisk: (optional) Array of total risk in percent units, corresponds to ordering of asset universe (null values not allowed)
#    - historicalBeta: (optional) Array of historical beta, corresponds to ordering of asset universe (null values not allowed)
# 3. Covariance Matrix
#    - 2D array of the covariance between the factors in daily variance units. Ordering corresponds to the ordering of the Factor Data list in payload
# 
# There are also some optional inputs:
# -  Issuer Specific Covariance: The covariance attributed to two assets being issued by the same company (also known as Linked Specific Risk)
#     - universeId1: Array of assets with issuer specific covariance to the asset in universeId2 at the same index. Each asset must also be present in the Asset Data universe
#     - universeId1: Array of assets with issuer specific covariance to the asset in universeId1 at the same index. Each asset must also be present in the Asset Data universe
#     - covariance: Array of the covariance between universeId1 and universeId2 at the same index. In daily variance units
# -  Factor Portfolios: The weights of assets in the universe that combine to provide exposure of 1 to each factor
#     - universe: Array of assets that make up the factor portfolios. Each asset must also be present in the Asset Data universe
#     - portfolio: Array of:
#                - factorId: Id of factor corresponding to the Factor Data factorIds
#                - weights: Array of weights of each asset id, corresponding to the ordering of the universe given. Must have a weight for each asset in the universe, can have weights equal to 0 (null values not allowed)
# 
# 
# 
# ### Upload Full Data

# In[ ]:


data = {
    'date': '2021-01-13',  # Note: You can only upload to dates in your risk model's calendar
    'assetData': {
        'universe': ['B02V2Q0', '6560713', 'B3Q15X5', '0709954'],
        'specificRisk': [12.09, 45.12, 3.09, 1.0],
        'factorExposure': [
            {'1': 0.23, '2': 0.023},
            {'1': 0.023, '2': 2.09, '3': 0.3},
            {'1': 0.063, '2': 2.069, '3': 0.73},
            {'2': 0.067, '3': 0.93}
        ],
        'historicalBeta': [0.12, 0.45, 1.2, 0.3]
    },
    'factorData': [
        {
            'factorId': '1',
            'factorName': 'USD',
            'factorCategory': 'Currency',
            'factorCategoryId': 'CUR',
            'factorReturn': 0.5
        },
        {
            'factorId': '2',
            'factorName': 'Technology',
            'factorCategory': 'Industry',
            'factorCategoryId': 'IND',
            'factorReturn': 0.3
        },
        {
            'factorId': '3',
            'factorName': 'Momentum',
            'factorCategory': 'Style',
            'factorCategoryId': 'RI',
            'factorReturn': 0.2
        }
    ],
    'covarianceMatrix': [
        [0.089, 0.0123, 0.345],
        [0.0123, 0.767, 0.045],
        [0.345, 0.045, 0.0987]
    ],
    'issuerSpecificCovariance': {
        'universeId1': ['B02V2Q0', '6560713'],
        'universeId2': ['B3Q15X5', '0709954'],
        'covariance': [0.03754, 0.01234]
    },
    'factorPortfolios': {
        'universe': ['B02V2Q0', '6560713'],
        'portfolio': [
            {
                'factorId': '2',
                'weights': [0.25, 0.75]
            },
            {
                'factorId': '3',
                'weights': [0.33, 0.66]
            },
            {
                'factorId': '1',
                'weights': [0.80, 0.20]
            }
        ]
    }
}

# use the flag mass-asset_batch_size to determine how many assets to upload at once if the data
# payload is large enough to require splitting apart and uploading separately. A typical max_asset_batch_size
# used by multiple vendors is 12000--if there are still frequent timeouts with this size try reducing the size
model.upload_data(data, max_asset_batch_size=1000)

# Check which days have data posted:
dates_posted = model.get_dates()


# ### Upload Partial Data
# 
# Users may also want to upload their data in separate stages. This is supported using the partial data upload function.
# Partial data must always include a date, and is combined with any data uploaded previously on that date. If repeat data is detected, the most recently uploaded data will replace the previously uploaded data.
# The target universe size describes the unique number of assets the API expects before it considers the data upload complete.
# 
# For example, we can update the previously uploaded Issuer Specific Covariance data with a target universe size of 3, indicating
# that after this post, we should consider the Issuer Specific Covariance data complete on this date

# In[ ]:


from gs_quant.models.risk_model import RiskModelEventType

partial_data =  {
    'date': '2021-01-13',
    'issuerSpecificCovariance': {
        'universeId1': ['BYVY8G0', '2073390'],
        'universeId2': ['BYY88Y7', 'BYVY8G0'],
        'covariance': [0.3754, 0.1234]
    }
}

model.upload_data(partial_data, max_asset_batch_size=10)

# Check which days have issuer specific covariance data posted:
isc_dates_posted = model.get_dates(event_type=RiskModelEventType.Risk_Model_ISC_Data)

# ## Make the Model Available on the Marquee UI
# 
# The next step is enabling your model to be visible through the Marquee web interface by updating the risk model's coverage dataset with the risk model asset universe.
# 

# In[ ]:


# The risk model will now appear in the dropdown on the "create portfolios" page once coverage is posted

model.upload_asset_coverage_data()

# ## Enhance Factor Descriptions and Tool Tips
# 
# The last optional step is adding tooltips and descriptions to the risk model factors. We highly encourage you to do this for every non-binary factor in your model (such as your style factors), so that Marquee UI users of your model can leverage the tooltips and descriptions to better understand how the factors were constructed and what they represent. 

# In[ ]:


from gs_quant.models.risk_model import FactorType, RiskModelFactor

identifier = '3'
tooltip = 'Short description that appears when you hover over the factor name on our UI.'
description = 'Longer description that appears on the portfolio drill-down page of this factor.'
glossary_description = 'Longest description to describe the factor in depth on our risk model glossary page.'

factor = RiskModelFactor(
    identifier=identifier,
    type_=FactorType.Factor,
    tooltip=tooltip,
    description=description,
    glossary_description=glossary_description
)

model.save_factor_metadata(factor)

# 
