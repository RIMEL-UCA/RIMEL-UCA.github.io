#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# First initialize GsSession; external users should substitute client_id and client_secret fields with their credentials
from gs_quant.session import GsSession, Environment
import seaborn as sns

GsSession.use(Environment.PROD, client_id=None, client_secret=None)

# Global parameters for plots
params = {'axes.axisbelow': False, 'axes.edgecolor': 'lightgrey', 'axes.grid': False, 'axes.labelcolor': 'dimgrey', 'axes.labelsize': 'large', 'axes.spines.right': False, 'axes.spines.top': False, 'axes.titlesize': 'x-large','axes.titleweight': 'bold',
    'figure.dpi': 200, 'figure.facecolor': 'white', 'figure.figsize': (15, 7), 'legend.frameon': False, 'legend.fontsize': 'xx-small', 'lines.solid_capstyle': 'round', 'patch.edgecolor': 'w', 'patch.force_edgecolor': True, 'text.color': 'dimgrey',
    'xtick.bottom': False, 'xtick.color': 'dimgrey', 'xtick.direction': 'out', 'xtick.labelsize': 'x-small', 'xtick.top': False, 'ytick.color': 'dimgrey', 'ytick.direction': 'out', 'ytick.left': False, 'ytick.right': False, 'ytick.labelsize': 'small'
}

exp_palette= ['#53aaea', '#67b4ec', '#7bbeef', '#8fc8f1', '#a4d2f4', '#b8dcf6', '#cbe6f9', '#dfeffb', '#fce4e1', '#fad6d2', '#f9c9c3', '#f7bbb4', '#f6ada5', '#f49f96', '#f39287', '#f18478']
longer_exp_palette = [col for col in exp_palette for i in range(3)]
sns.set(style='white', rc=params)

# # Discover your Macro risk with Quant Insight models
# 
# Quant Insight's Macro risk models provide an insight in the relationship between movement in an asset price and
# macroeconomic factors such as economic growth, monetary policy and commodity prices. The goal is to understand how much
# of the movement in the asset price is attributable to those macroeconomic factors. The GS quant class `MacroRiskModel`
# provides an array of functions that query macro risk model data. In this tutorial, we will look at querying all the
# available macro risk model data.
# 
# Currently, the macro risk models that are available for programmatic access are below:
# 
# | Risk Model Name             | Risk Model Id |
# |-----------------------------|---------------|
# | US Equity Model (Long Term) | QI_US_EQUITY_LT |
# | EU Equity Model (Long Term) | QI_EU_EQUITY_LT |
# | UK Equity Model (Long Term) | QI_UK_EQUITY_LT |
# | APAC Equity Model (Long Term) | QI_APAC_EQUITY_LT |
# 
# ## Macro Factor Data
# 
# Macro Factors are grouped within a factor category.
# Here are some examples of macro factor categories along with a list of macro factors in that category:
# 
# | Macro Factor Category | Macro Factors                                              |
# |-----------------------|------------------------------------------------------------|
# | Inflation             | US 5Y Infl. Expec., US 2Y Infl. Expec., US 10Y Infl. Expec.|
# | Economic Growth       | Japan GDP, Euro GDP, China GDP                             |
# | CB Rate Expectations  | Fed Rate Expectations                                      |
# | Energy                | WTI                                                        |
# | Risk Aversion         | VDAX, VIX, VXEEM, Gold Silver Ratio                        |
# 
# For more macro factor categories and their descriptions, see the factor glossary.
# 
# #### Get All Available Macro Factor Categories
# 
# We can leverage `get_factor_data` in the `MacroRiskModel` class to get all the available macro factor categories
# in the model.
# 

# In[ ]:


from gs_quant.models.risk_model import MacroRiskModel, FactorType
import datetime as dt

start_date = dt.date(2022, 4, 1)

model_id = "QI_US_EQUITY_LT"
model = MacroRiskModel.get(model_id)

# Get the factors in dataframe with name, and type
factor_category_data = model.get_factor_data(start_date=start_date, factor_type=FactorType.Factor)
factor_category_data_reshaped = (
    factor_category_data.set_index("factorCategoryId")
                        .drop(columns={"name", "type", "identifier"})
                        .drop_duplicates()
                        .rename_axis("Factor Category Id")
                        .rename(columns={"factorCategory": "Factor Category"})
)
display(factor_category_data_reshaped)


# #### Get All Available Macro Factors
# 
# Within each macro factor category, we have several macro factors that are grouped together.

# In[ ]:


from gs_quant.models.risk_model import MacroRiskModel, FactorType
import datetime as dt
import pandas as pd

start_date = dt.date(2022, 4, 1)

model_id = "QI_US_EQUITY_LT"
model = MacroRiskModel.get(model_id)

# Get the factors in dataframe with name, and type
factor_data = model.get_factor_data(start_date=start_date, factor_type=FactorType.Factor)
factor_data_reshaped = (
    factor_data.rename(columns={"name": "Factor", "factorCategory": "Factor Category"})
               .sort_values(by=["Factor Category"])
               .drop(columns={"type", "factorCategoryId", "identifier"})
               .set_index("Factor Category")
               .stack().to_frame().droplevel(level=1).rename(columns={0: "Factor"})

)
display(factor_data_reshaped)

# Further, for a more granular view, we can get all the macro factors that are grouped within a factor category.
# Below is a list of all macro factors in the factorCategory "Risk Aversion".

# In[ ]:


macro_factors_in_risk_aversion = factor_data_reshaped.groupby("Factor Category").get_group("Risk Aversion")
display(macro_factors_in_risk_aversion)

# ## Asset Data
# 
# The metrics below are instrumental in providing insight in the relationship between movement of asset price and macro
# factors. These key metrics are outlined below:
# 
# | Metric                    | Description    |
# |----------------------------|---------------|
# | `Universe Macro Factor Sensitivity` | Percentage change in asset price for 1 standard deviation move up in the macro factor. |
# | `R Squared (Model Confidence)`               | Gauge of how sensitive the asset is to macroeconomic forces.  Typically values above 65% are considered to show strong explanatory power or ‘confidence’. |
# | `Fair Value Gap`          | The difference between the actual price of an asset and the Qi Model Value.  This is quoted both in absolute (in percentage) and in standard deviation terms.  |
# | `Specific Risk`            | Annualized asset volatility. |
# 
# 
# ## Get  your Portfolio Exposure to Macro Factors and Categories
# 
# Given a portfolio, we can get its cash exposure to a 1 standard deviation move up in each macro factor. The exposures are
# calculated by multiplying each asset notional amount to their sensitivity to each factor, which gives us individual asset
#  exposure to each macro factor. We then aggregate per asset exposures to get portfolio cash exposure to each macro \
#  factor.
# We can leverage the function `get_macro_exposure_table` of the `PortfolioManager` class. Note that exposure is expressed
# in the currency of the notional value of the portfolio.
# 
# We can get portfolio exposure to both macro factor categories and macro factors. We can thus gain an insight
# in which factor categories and macro factors within these categories are driving the portfolio.
# 
# #### Exposure to Macro Factor Categories
# 
# We need to pass the following parameters to the function:
# * `macro_risk_model`: The model to base your exposure data on
# * `date`: date for which to get exposure
# * `group_by_factor_category`: whether to get exposure per factor category.
# 
# The result will be sorted from top positive drivers to top negative drivers
# 
# Note that, in the result, the macro factor categories are ranked from most positive exposure to most negative portfolio exposure
# 
# Below:
# * We get asset level exposure to each factor category.
# * The last row of the resulting dataframe is the portfolio exposure to each factor category (asset level exposure aggregated).

# In[ ]:


from gs_quant.models.risk_model import MacroRiskModel, FactorType
from gs_quant.markets.portfolio_manager import PortfolioManager
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# Get the model
model = MacroRiskModel.get("QI_US_EQUITY_LT")

# Input the portfolio id
portfolio_id = "MPE2KR4BC4RKQ5KE"
pm = PortfolioManager(portfolio_id)
date = dt.date(2022, 5, 2)

# Get Per Asset Factor Category Exposure Dataframe
exposure_factor_category_df = pm.get_macro_exposure(model=model, date=date, factor_type=FactorType.Category)
exposure_factor_category_df.loc["Total Factor Category Exposure", "Asset Name"] = "Total"
exposure_factor_category_df = exposure_factor_category_df.set_index("Asset Name", drop=True)

# Get the Portfolio Total
total = exposure_factor_category_df.loc["Total", :].to_frame().drop("Notional")

# Display the Dataframes.
styles = [{'selector': "caption", 'props': [("font-size", "200%"), ("font-weight", "bold"), ("color", 'black')]}, {'selector': 'thead', 'props': [("background-color", "silver"), ("font-size", "13px")]}, {'selector': 'tr', 'props': [("background-color", "whitesmoke"), ("font-size", "13px"), ("border", "solid"), ("border-width", "0.001em")]},
          {'selector': 'td', 'props': [("background-color", "aliceblue"), ("font-size", "13px")]}]
exposure_styler = (
    exposure_factor_category_df.style.set_caption("Portfolio Cash Exposure to Factor Categories")
                                     .format('${:,.0f}')
                                     .applymap(lambda v: "color:red;" if v < 0 else None)
                                     .set_table_styles(styles)
)
styles.pop(0)
total_styler = total.style.format('${:,.0f}').applymap(lambda v: "color:red;" if v < 0 else None).set_table_styles(styles)
display(exposure_styler)
display(total_styler)

# Plot
fig, ax = plt.subplots(constrained_layout=True)
sns.barplot(data=total.reset_index(), x="Factor Category", y='Total', ax=ax, palette=exp_palette)
ax.set_xlabel("Factor Category")
ax.set_ylabel("Portfolio Notional Change for 1 std Move in the Factor Category", fontsize='medium')
ax.set_title("Portfolio Exposure to Factor Categories")
ax.tick_params(axis='x', labelrotation = 50)
plt.show()


# #### Exposure to Macro Factors
# Once we get portfolio exposure to factor categories, we can get a granular view of the portfolio exposure to individual macro factors within that factor category.
# 
# Here are the user inputs:
# * `model`: The model we are using.
# * `factor_categories`: A list of factor categories whose factors we want exposure for. If empty, it will default to returning exposure to the top 2 positive and negative macro drivers
# * `portfolio_id`: The portfolio we are getting macro exposure for.
# * `date`: Date 

# In[ ]:


from gs_quant.models.risk_model import MacroRiskModel, FactorType
from gs_quant.markets.portfolio_manager import PortfolioManager
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# model id
model_id = "QI_US_EQUITY_LT"
model = MacroRiskModel.get(model_id)

portfolio_id = "MPE2KR4BC4RKQ5KE"
pm = PortfolioManager(portfolio_id)

# date
date = dt.date(2022, 5, 2)

# Add Factor categories whose factors you want exposure for
factor_categories = []
factor_categories_data = model.get_many_factors(start_date=date, end_date=date, factor_names=factor_categories, factor_type=FactorType.Category)

# Get Per Asset Factor Exposure dataframe
exp_factor_df = pm.get_macro_exposure(model=model, date=date, factor_type=FactorType.Factor, factor_categories=factor_categories_data)
exp_factor_df.loc["Total Factor Exposure", ("Asset Information", "Asset Name")] = "Total"
exp_factor_df = exp_factor_df.set_index(("Asset Information", "Asset Name"), drop=True)

# Total Portfolio Exposure
total = exp_factor_df.loc["Total", :].to_frame().drop(("Asset Information", "Notional"))

# Display the Dataframes.
styles = [{'selector': "caption", 'props': [("font-size", "200%"), ("font-weight", "bold"), ("color", 'black')]}, {'selector': 'thead', 'props': [("background-color", "silver"), ("font-size", "13px")]}, {'selector': 'tr', 'props': [("background-color", "whitesmoke"), ("font-size", "13px"), ("border", "solid"), ("border-width", "0.001em")]},
          {'selector': 'td', 'props': [("background-color", "aliceblue"), ("font-size", "13px")]}]
exposure_styler = (
    exp_factor_df.style.set_caption("Portfolio Cash Exposure to Macro Factors")
                       .format('${:,.0f}')
                       .applymap(lambda v: "color:red;" if v < 0 else None).set_table_styles(styles)
)
styles.pop(0)
total_styler = total.style.format('${:,.0f}').applymap(lambda v: "color:red;" if v < 0 else None).set_table_styles(styles)
display(exposure_styler)
display(total_styler)

# Plot the portfolio percent change if every  factor category shifted 1 std
fig, ax = plt.subplots(constrained_layout=True)
sns.barplot(data=total.sort_values(by=["Total"], ascending=False).reset_index(), x="Factor", y='Total', ax=ax, palette=longer_exp_palette)
ax.set_xlabel("Macro Factor")
ax.set_ylabel("Portfolio Notional Change for 1 std Move in the Macro Factor", fontsize='medium')
ax.set_title("Portfolio Exposure to Macro Factors")
ax.tick_params(axis='x', labelrotation=60)
plt.show()

# 
# ### Run empirical stress tests to observe impact on portfolio
# 
# We can bump up or down any combination of macro factors and see the impact on the portfolio. Since your portfolio's
# sensitivities to each macro factor are independent of each other, we can isolate a (or a combination of) macro factor categories and factors,
# and assign a shift up or down in standard deviation and observe the change in the portfolio notional value.
# 
# Below, we define a class `MacroScenario` that takes in a date and a dictionary of macro factor categories or macro factors and their suggested standard
# deviation shifts.

# In[ ]:


from gs_quant.session import GsSession, Environment
from gs_quant.models.risk_model import MacroRiskModel, FactorType
from gs_quant.markets.factor import Factor
from gs_quant.markets.portfolio_manager import PortfolioManager
from typing import Dict
import datetime as dt
import pandas as pd
import numpy as np

GsSession.use(Environment.PROD, client_id=None, client_secret=None)

class MacroScenario:
    def __init__(self, date: dt.date, factor_std_shifts: Dict[Factor, float]):
        self.date = date
        self.factor_std_shifts = factor_std_shifts


def macro_stress_test(exposure_df: pd.DataFrame,
                      scenario: MacroScenario,
                      is_factor_category_stress: bool) ->  pd.DataFrame:

    """
    For a given portfolio, find how shifts in standard deviation in the selected macro factors affects
    the portfolio's notional value.
    :param exposure_df: exposure of the portfolio to the requested macro factors/factor categories
    :param scenario: scenario to stress test for. Contains date and a dict of factors and respective shifts in standard
    deviation
    :is_factor_category_stress: whether we are stress testing for shifts in factor categories or shifts in factors.
    :return: portfolio notional change and change in portfolio exposure for each macro factor
    """

    factors_and_std = scenario.factor_std_shifts
    factors_to_shift = {factor.name: factor.category for factor in factors_and_std.keys()}
    total_row_name = "Total Factor Category Exposure" if is_factor_category_stress else "Total Factor Exposure"
    notional_column_name = "Notional" if is_factor_category_stress else ("Asset Information", "Notional")
    portfolio_notional = exposure_df.loc[total_row_name, notional_column_name]
    exposure_df = exposure_df[list(factors_to_shift.keys())] if is_factor_category_stress else\
        exposure_df[list(zip(factors_to_shift.values(), factors_to_shift.keys()))]

    fact_name = 'Factor Category' if is_factor_category_stress else 'Factor'
    col_1_name = f"Exposure to {fact_name} Before Stress Test"
    col_2_name = f"Exposure to {fact_name} After Stress Test"
    portfolio_change_df = pd.DataFrame(np.zeros((4, len(set(exposure_df.columns.tolist())))),
                                       index=["Standard Deviation Shift", col_1_name, col_2_name,
                                              "Percent Change After Stress Test"],
                                       columns=set(exposure_df.columns.tolist()))

    for factor, std_shift in factors_and_std.items():
        factor_name = factor.name if is_factor_category_stress else (factors_to_shift[factor.name], factor.name)
        old_exposure = exposure_df.loc[total_row_name, factor_name]
        new_exposure = old_exposure * std_shift
        portfolio_change_df.loc["Standard Deviation Shift", factor_name] = std_shift
        portfolio_change_df.loc[col_1_name, factor_name] = old_exposure
        portfolio_change_df.loc[col_2_name, factor_name] = new_exposure
        portfolio_change_df.loc["Percent Change After Stress Test", factor_name] = (new_exposure / portfolio_notional) * 100

    total_name = "Total" if is_factor_category_stress else ("Total", "Total")
    total_sum = portfolio_change_df.agg(np.sum, axis="columns").to_frame().rename(columns={0: total_name})

    portfolio_change_df = pd.concat([portfolio_change_df, total_sum], axis='columns')
    portfolio_change_df.loc["Standard Deviation Shift", total_name] = np.NAN

    return portfolio_change_df

# ### Factor Standard Deviation
# 
# Before the stress test, we can first determine, for each factor, the value of one shift in standard deviation in absolute terms. We can leverage the function `get_factor_standard_deviation` of the `MacroRiskModel` class.

# In[ ]:


from gs_quant.models.risk_model import MacroRiskModel
import datetime as dt
import numpy as np
import pandas as pd

start_date = dt.date(2022, 1, 1)
end_date = dt.date(2022, 5, 2)

date = model.get_dates(start_date, end_date)[-1]

model_id = "QI_US_EQUITY_LT"
model = MacroRiskModel.get(model_id)

factor_category_shift_std = {
   "Inflation": 3,
   "Real Rates": 3,
   "Economic Growth": -3
}

factor_data = model.get_factor_data(date, date)
factor_data = factor_data.loc[factor_data['factorCategory'].isin(list(factor_category_shift_std.keys())), :]
factor_data = factor_data.set_index("name")

factor_std_df = model.get_factor_standard_deviation(start_date=date, end_date=date, factors=factor_data.index.tolist())

factor_std_df = (
    factor_std_df.set_axis(pd.MultiIndex.from_tuples([(factor_data.loc[f, 'factorCategory'], f) for f in factor_std_df.columns.values]), axis=1)
                 .rename_axis(("Factor Category", "Factor"), axis=1)
                 .sort_index(level=0, axis=1)
                 .round(3)
                 .T.rename(columns={date.isoformat(): "Value of 1 STD"})
)

display((
    factor_std_df.style.set_caption(f"One Standard Deviation of Factors on {date.isoformat()}")
                       .format("{:.3f}", na_rep="N/A").background_gradient()
                       .set_table_styles([dict(selector="caption", props=[("font-size", "120%"), ("font-weight", "bold"),("color", 'black')])])
))

# #### Stress Test Shifts in Factor Categories
# 
# We can stress test for shifts in factor categories and observe the impact on our portfolio. 
# 
# Here are the user inputs:
# * `model_id`: The model we are using
# * `portfolio_id`: Portfolio we are stress testing for.
# * `factors_shifts_std`: Map of factor categories and their standard deviation scenario shift

# In[ ]:


from gs_quant.models.risk_model import MacroRiskModel
import datetime as dt
import numpy as np
import pandas as pd

date = dt.date(2022, 5, 2)

model_id = "QI_US_EQUITY_LT"
model = MacroRiskModel.get(model_id)

portfolio_id = "MPE2KR4BC4RKQ5KE"
pm = PortfolioManager(portfolio_id)

factor_category_shift_std = {
   "Inflation": 0.5,
   "Real Rates": 3,
   "Economic Growth": -2
}
many_factor_categories = model.get_many_factors(start_date=date, end_date=date,
                                                factor_names=list(factor_category_shift_std.keys()),
                                                factor_type=FactorType.Category)
std_dict = {factor: factor_category_shift_std[factor.name] for factor in many_factor_categories}

# Create a MacroScenario object
scenario = MacroScenario(date, std_dict)

# First, get the portfolio factor category exposure
exposure_df = pm.get_macro_exposure(model=model, date=scenario.date, factor_type=FactorType.Category, factor_categories=many_factor_categories)

# Then calculate portfolio change given std shifts in the factor categories in `scenario`
portfolio_change = macro_stress_test(exposure_df, scenario, is_factor_category_stress=True).T.rename_axis("Factor Category")

# Display
styles = [dict(selector="caption", props=[("font-size", "120%"), ("font-weight", "bold"),("color", 'black')])]
display((
    portfolio_change.style.set_caption("Stress Test: Shifts in Factor Categories and their Impact on Portfolio")
                          .format({"Percent Change After Stress Test": "{:.3f}%", "Exposure to Factor Category Before Stress Test": "${:,.0f}", "Exposure to Factor Category After Stress Test": "${:,.0f}", "Standard Deviation Shift": "{:.2f}"}, na_rep="N/A")
                          .background_gradient().set_table_styles(styles))
)


# #### Stress Test Shifts in Factors
# 
# We can also stress test for shifts in individual factors and observe the impact on our portfolio. 
# 
# Here are the user inputs:
# * `model_id`: The model we are using
# * `portfolio_id`: Portfolio we are stress testing for.
# * `factors_shifts_std`: Map of factors and their standard deviation scenario shifts

# In[ ]:


from gs_quant.models.risk_model import MacroRiskModel
from gs_quant.models.risk_model import DataAssetsRequest, RiskModelUniverseIdentifierRequest as UniverseIdentifier, FactorType
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

date = dt.date(2022, 5, 2)

model_id = "QI_US_EQUITY_LT"
model = MacroRiskModel.get(model_id)

portfolio_id = "MPE2KR4BC4RKQ5KE"
pm = PortfolioManager(portfolio_id)

factors_shift_std = {
    "US 2Y Infl. Expec.": 0.5,
    "US 5Y Infl. Expec.": 0.5,
    "US 10Y Infl. Expec.": 0.5,
    "USD 10Y Real Rate": 3.0,
    "JPY 10Y Real Rate": 3.0,
    "EUR 10Y Real Rate": 3.0
}

many_factors = model.get_many_factors(start_date=date, end_date=date, factor_names=list(factors_shift_std.keys()),
                                      factor_type=FactorType.Factor)
std_dict = {factor: factors_shift_std[factor.name] for factor in many_factors}

# MacroScenario object
scenario = MacroScenario(date, std_dict)

# First, get the portfolio factor exposure
exposure_df = pm.get_macro_exposure(model=model, date=scenario.date, factor_type=FactorType.Factor).sort_values(by=["Total Factor Exposure"], axis=1, ascending=False)

# Then calculate portfolio change given std shifts in the factors in `scenario`
portfolio_change = macro_stress_test(exposure_df, scenario, is_factor_category_stress=False).T.rename_axis(["Factor Category", "Factor"])

# Display
styles = [dict(selector="caption", props=[("font-size", "130%"), ("font-weight", "bold"),("color", 'black')])]
display((
    portfolio_change.style.set_caption("Stress Test: Shifts in Factors and their Impact on Portfolio")
                          .format({"Percent Change After Stress Test": "{:.3f}%", "Exposure to Factor Before Stress Test": "${:,.0f}", "Exposure to Factor After Stress Test": "${:,.0f}", "Standard Deviation Shift": "{:.2f}"}, na_rep="N/A")
                          .background_gradient().set_table_styles(styles))
)

# ## Which assets in the portfolio are in a Macro Regime?
# 
# In a macro regime, macro factors are the most significant in explaining the movement in asset price. To determine which
# assets are in a macro regime, we look at the R Squared value, the proportion of the movement in asset price that is explained
# by the macro factors. Assets with R Squared values above 65% are in a macro regime while values below 65% are driven
# primarily by micro and idiosyncratic risk.
# 
# Once an asset is in a macro regime, we can look at the fair value gap to determine if it is rich or cheap.
# The fair value gap metric is the difference between the spot price of an asset and the asset model value. An asset is
#  undervalued when its fair value gap is negative while it is overvalued when it is a positive value.
# 
# Given a portfolio, we can find out how many assets are in a macro regime at a specific date. We can leverage the function
# `get_fair_value_gap` and `get_r_squared` to determine the regime of each asset in the portfolio as well as its fair value
# gap. Note that the fair value gap data is available both in absolute terms (in percentage) and in standard deviation terms.
# 
# Here are the user inputs:
# * `portfolio_id`: Portfolio we are using. 
# * `model_id`: The model we are using to get r_squared and fair value gap data.

# In[ ]:


from gs_quant.models.risk_model import MacroRiskModel, DataAssetsRequest as DataRequest, \
    RiskModelUniverseIdentifierRequest, Unit
from gs_quant.markets.portfolio_manager import PortfolioManager
from IPython.display import display
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt

GsSession.use(Environment.PROD)

# Time range
start_date = dt.date(2022, 5, 2)
end_date = dt.date(2022, 5, 2)

# Model
model_id = "QI_US_EQUITY_LT"
model = MacroRiskModel.get(model_id)

# Portfolio and positions
portfolio_id = "MPE2KR4BC4RKQ5KE"
pm = PortfolioManager(portfolio_id)
positions = pm.get_latest_position_set().positions

universe = [position.identifier for position in positions]
universe_for_request = DataRequest(RiskModelUniverseIdentifierRequest.bbid, universe)

# Get fair value gap and r squared for the assets in the portfolio
fvg_std_df = model.get_fair_value_gap(start_date=start_date, end_date=end_date,
                                      fair_value_gap_unit=Unit.STANDARD_DEVIATION, assets=universe_for_request)
r_squared_df = model.get_r_squared(start_date=start_date, end_date=end_date, assets=universe_for_request)

date = dt.date(2022, 5, 2).strftime("%Y-%m-%d")
fvg_one_date_df = fvg_std_df.loc[date, :].to_frame().rename(columns={date: "Fair Value Gap"})
r_squared_one_date_df = r_squared_df.loc[date, :].to_frame().rename(columns={date: "R Squared"})
fvg_rsq_df = r_squared_one_date_df.join(fvg_one_date_df).reset_index().round(2)

# Get the Macro cheap assets
macro_cheap_df = (
    fvg_rsq_df[(fvg_rsq_df["R Squared"] > 65) & ((fvg_rsq_df["Fair Value Gap"] < 0))].set_index("index")
                                                                                     .rename_axis("Asset")
                                                                                     .sort_values(by="Fair Value Gap", ascending=True)
)

# Plot Quadrant of assets that are macro cheap/rich and micro
fig, ax = plt.subplots(figsize=(20, 15), constrained_layout=True)
sns.scatterplot(data=fvg_rsq_df, x="R Squared", y='Fair Value Gap', hue='index', legend=False)
ax.set_title(f"Macro Rich vs Macro Cheap Assets on {date}")
ax.set_xlabel("R Squared (Model Confidence) in %")
ax.set_ylabel("Fair Value Gap (sigma)")
ax.set_xlim(0, 100)
ax.set_ylim(-3, 3)

def annotate_plot(data, ax, index):
    for i in range(data.shape[0]):
        if data[index][i] == 'DHR UN':
            ax.text(data["R Squared"][i], data["Fair Value Gap"][i], s=data[index][i], size='x-large', color='r')
        else:
            ax.text(data["R Squared"][i], data["Fair Value Gap"][i], s=data[index][i], size='large')

annotate_plot(fvg_rsq_df, ax, "index")

for pair, label in {(75, 2.2): "Macro Rich", (25, 2.2): "Micro Regime", (25, -2.2): "Micro Regime", (72, -2.2): "Macro Cheap"}.items():
    ax.text(x=pair[0], y=pair[1], s=label, alpha=0.7, fontsize='xx-large', color='g', fontweight='bold')
ax.axhline(y=0, color='k', linestyle='dashed')
ax.axvline(x=65, color='k', linestyle='dashed')

# Plot Just macro cheap assets
fig, ax = plt.subplots(figsize=(18, 7), constrained_layout=True)
sns.scatterplot(data=macro_cheap_df.reset_index(), x="R Squared", y="Fair Value Gap", hue="Asset", legend=False)
annotate_plot(macro_cheap_df.reset_index(), ax, "Asset")
ax.set_title(f"Macro Cheap Assets on {end_date}")
ax.set_ylabel("Fair Value Gap")
ax.set_xlabel("R Squared (in %)")
ax.text(x=75, y=-2.0, s="Macro Cheap", alpha=0.7, fontsize='xx-large', color='g', fontweight='bold')

plt.show()

# The first plot above classifies assets into 3 groups: macro cheap, macro rich and assets in a micro regime. The second plot shows the bottom right of the first
# plot, i.e all the assets that are macro cheap. 

# 
# ### Asset Sensitivity to Factors
# 
# The asset Sensitivity to a macro factor is the percentage change in the asset price for 1 standard deviation move
# in that macro factor. We can leverage the function `get_universe_sensitivity` of the `MacroRiskModel` class to get a
# time series of daily asset sensitivity to macro factors for a list of assets.
# 
# Here are the parameters that are needed:
# * `start_date` and `end_date`: Time range for which to get sensitivity for. 
# * `model_id`: The model we are using.
# * `asset`: Asset to get macro sensitivity for.
# 
# Below, we are querying a time series of daily sensitivity data over the time range for one of the macro cheap assets in the portfolio. 

# #### Asset Sensitivity to Factor Categories

# In[ ]:


from gs_quant.models.risk_model import MacroRiskModel, DataAssetsRequest, RiskModelUniverseIdentifierRequest as UniverseIdentifier, FactorType
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Time range
start_date = dt.date(2022, 5, 2)
end_date = dt.date(2022, 5, 2)

# The model to get sensitivity data for
model_id = "QI_US_EQUITY_LT"
model = MacroRiskModel.get(model_id)

# Universe
asset = "DHR UN"
universe = [asset]
universe_for_request = DataAssetsRequest(UniverseIdentifier.bbid, universe)

# Get asset sensitivity to factor categories
factor_sens_df = model.get_universe_sensitivity(start_date=start_date, end_date=end_date, assets=universe_for_request,
                                                factor_type=FactorType.Category, get_factors_by_name=True)
factor_sens_df = (
    factor_sens_df.droplevel(1)
                  .sort_values(by=[asset], ascending=False, axis=1)
                  .rename_axis("Factor Category", axis=1)
)

# Display the dataframe
factor_sens_to_display = factor_sens_df.T.rename(columns={asset: f"Percent Change in Asset Price for 1 STD shift in each Factor Category for {asset}"})
display((
    factor_sens_to_display.style.set_caption(f"Sensitivity of {asset} to Factor Categories on {start_date}")
                          .format("{:.2f}%").applymap(lambda v: "color:red;" if v < 0 else None)
                          .set_table_styles([{'selector': "caption", 'props': [("font-size", "100%"), ("font-weight", "bold"), ("color", 'black')]},
                                             {'selector': 'table', 'props': [('width', '100px')]},
                                             {'selector': 'thead', 'props': [("background-color", "silver"), ("font-size", "12px"), ("width", "400px")]},
                                             {'selector': 'tr', 'props': [("background-color", "gainsboro"), ("font-size", "12px"), ("border", "solid"),
                                                                       ("border-width", "1px")]},
                                             {'selector': 'td', 'props': [("background-color", "aliceblue"), ("font-size", "12px")]}]))
)

# Plot the portfolio percent change if every  factor category shifted 1 std
fig, ax = plt.subplots(constrained_layout=True)
sns.barplot(data=factor_sens_df.T.reset_index(), x="Factor Category", y=asset, ax=ax, palette=exp_palette)
ax.set_title(f"Expected % change in {asset} Price for 1 std Move in Factor Category on {end_date}")
ax.set_xlabel("Factor Category")
ax.set_ylabel("")
ax.set_yticks([], [])
ax.tick_params(axis='x', labelrotation= 30)

labls = factor_sens_df.T.reset_index()["Factor Category"].values.tolist()
values = factor_sens_df.T.reset_index()[asset].values.tolist()
for i in range(len(labls)):
    if values[i] > 0:
        ax.text(i, values[i] + 0.01, s=f'{round(values[i], 1)}%', ha='center', fontsize='x-small')
    else:
        ax.text(i, values[i] - 0.08, s=f'{round(values[i], 1)}%', ha='center', fontsize='x-small')

sns.despine(left=True)
plt.show()

# #### Asset Sensitivity to Factors
# 
# Once we know an asset sensitivity to factor categories, we can also query its sensitivity to individual macro factors within a factor category.
# 
# Here are the user inputs:
# * `model`: The model we are using.
# * `factor_categories`: A list of factor categories whose factors we want sensitivity for. If empty, it will default to returning sensitivity to the top 2 positive and negative macro drivers
# * `start_date` and `end_date`: Time range to get sensitivity data for an asset

# In[ ]:


from gs_quant.models.risk_model import MacroRiskModel
from gs_quant.models.risk_model import DataAssetsRequest, RiskModelUniverseIdentifierRequest as UniverseIdentifier, FactorType
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd

# Time range
start_date = dt.date(2022, 5, 2)
end_date = dt.date(2022, 5, 2)

# The model to get data for
model_id = "QI_US_EQUITY_LT"
model = MacroRiskModel.get(model_id)

# Select factor_categories
factor_categories = []

# Universe
asset = "DHR UN"
universe = [asset]
universe_for_request = DataAssetsRequest(UniverseIdentifier.bbid, universe)
factor_data = model.get_factor_data(start_date=start_date, end_date=end_date).set_index('name')

# Get asset sensitivity to factors as a dataframe
factor_sens_df = model.get_universe_sensitivity(start_date=start_date, end_date=end_date,
                                                assets=universe_for_request, get_factors_by_name=True)
factor_sens_df = factor_sens_df.set_axis(pd.MultiIndex.from_tuples(
        [(factor_data.loc[x, 'factorCategory'], x) for x in factor_sens_df.columns]), axis=1).droplevel(1)

if not factor_categories:
    # Get top drivers
    top_drivers = (
        model.get_universe_sensitivity(start_date=start_date, end_date=end_date, assets=universe_for_request,
                                       factor_type=FactorType.Category,
                                       get_factors_by_name=True).droplevel(1)
                                                                .sort_values(by=asset, axis=1, ascending=False)
                                                                .columns.values.tolist()
    )
    factor_categories = top_drivers[0:2] + top_drivers[-2:None]

factor_sens_df = (
    factor_sens_df[factor_categories].T.rename_axis(["Factor Category", "Factor"])
                                       .sort_values(by=asset, ascending=False)
                                       .rename(columns={asset: f"Percent Change in Asset Price for 1 STD shift in each Factor for {asset}"})
)

# Display the dataframe
display((
    factor_sens_df.style.set_caption(f"Sensitivity of {asset} to Factors on {start_date}")
                        .format("{:.2f}%")
                        .applymap(lambda v: "color:red;" if v < 0 else None)
                        .set_table_styles([{'selector': "caption", 'props': [("font-size", "100%"), ("font-weight", "bold"), ("color", 'black')]},
                                           {'selector': 'table', 'props': [('width', '100px')]},
                                           {'selector': 'thead', 'props': [("background-color", "silver"), ("font-size", "12px"), ("width", "400px")]},
                                           {'selector': 'tr', 'props': [("background-color", "gainsboro"), ("font-size", "12px"), ("border", "solid"),
                                                                       ("border-width", "1px")]},
                                           {'selector': 'td', 'props': [("background-color", "aliceblue"), ("font-size", "12px")]}]))
)

# Plot the portfolio percent change if every  factor category shifted 1 std
fig, ax = plt.subplots(constrained_layout=True)
sns.barplot(data=factor_sens_df.reset_index(), x="Factor", y=f"Percent Change in Asset Price for 1 STD shift in each Factor for {asset}", ax=ax, palette=exp_palette)
ax.set_title(f"Expected % change in {asset} Price for 1 std Move in Factor Category on {end_date}")
ax.set_xlabel("Macro Factor")
ax.set_ylabel("")
ax.set_yticks([], [])
ax.tick_params(axis='x', labelrotation=60)

labls = factor_sens_df.reset_index()["Factor"].values.tolist()
values = factor_sens_df.reset_index()[f"Percent Change in Asset Price for 1 STD shift in each Factor for {asset}"].values.tolist()
for i in range(len(labls)):
    if values[i] > 0:
        ax.text(i, values[i] + 0.01, s=f'{round(values[i], 1)}%', ha='center', fontsize='x-small')
    else:
        ax.text(i, values[i] - 0.03, s=f'{round(values[i], 1)}%', ha='center', fontsize='x-small')

sns.despine(left=True)
plt.show()

# ### Historical Asset Sensitivity to Factor Categories
# 
# We can query asset sensitivity to factor categories over time. To get sensitivity to factor categories, make sure that the factor_type parameter in the function `get_universe_sensitivity` is set to `FactorType.Category`.
# 
# If the factor_categories parameter is not specified, the top drivers of the asset will be shown instead.
# 
# On the plot, she shaded area represents the time range when the asset was in a macro regime.

# In[ ]:


from gs_quant.models.risk_model import MacroRiskModel
from gs_quant.models.risk_model import DataAssetsRequest, RiskModelUniverseIdentifierRequest as UniverseIdentifier, FactorType
import datetime as dt
import matplotlib.pyplot as plt
from itertools import groupby, cycle

# Time range
start_date = dt.date(2021, 1, 1)
end_date = dt.date(2022, 5, 2)

# The model to get data for
model_id = "QI_US_EQUITY_LT"
model = MacroRiskModel.get(model_id)

# Asset and factor categories
asset = "DHR UN"
factor_categories = []

universe = [asset]
universe_for_request = DataAssetsRequest(UniverseIdentifier.bbid, universe)

# Get asset sensitivity to factor category (note that factor_type is set to `Category`)
factor_category_sens_df = model.get_universe_sensitivity(start_date=start_date, end_date=end_date, assets=universe_for_request,
                                                         factor_type=FactorType.Category, get_factors_by_name=True)

# If no factor categories passed in, get the top drivers as of end_date
if not factor_categories:
    top_drivers = (
        factor_category_sens_df.loc[(asset, end_date.strftime("%Y-%m-%d")), :].to_frame()
                                                                              .droplevel(1, axis=1)
                                                                              .sort_values(by=asset, ascending=False)
                                                                              .index.values.tolist()
    )
    factor_categories = top_drivers[0:2] + top_drivers[-2:None]
factor_category_sens_df = factor_category_sens_df[factor_categories].droplevel(0)

# Get date range when `asset` was in a macro regime
r_squared_historical = (
    model.get_r_squared(start_date=start_date, end_date=end_date,
                        assets=DataAssetsRequest(UniverseIdentifier.bbid, [asset])).reset_index()
                                                                                   .rename(columns={"index": "Date"})
)
macro_regime_dates = r_squared_historical.loc[r_squared_historical[asset] > 65].index.tolist()
temp_list = cycle(macro_regime_dates)
next(temp_list)
groups = groupby(macro_regime_dates, key=lambda j: j + 1 == next(temp_list))
macro_regime_date_range = [list(v) + [next(next(groups)[1])] for k, v in groups if k]

fig, ax = plt.subplots(constrained_layout=True)
factor_category_sens_df.reset_index().rename(columns={"index": "Date"}).plot(ax=ax, x="Date")
ax.set_title(f"Sensitivity of {universe[0]} to Factor Categories on {end_date} Over Time")
ax.set_ylabel("Percentage change in asset price for 1 std shift")
ax.axhline(y=0, linestyle='dashed')
for date_range in macro_regime_date_range:
    ax.axvspan(date_range[0], date_range[-1], color='aliceblue')
plt.show()

# ### Historical Asset Sensitivity to Factors
# 
# We can query asset sensitivity to indiviual factors within a factor category over time. The `factor_type` parameter in the function `get_universe_sensitivity` is by default set to `FactorType.Factor` to specify that we want exposure to individual factors.
# 
# If the `factors` parameter is not specified, the factors to which the asset has the most exposure will be returned. On the plot, the shaded area represents a time period when the asset was in a macro regime.

# In[ ]:


from gs_quant.models.risk_model import MacroRiskModel
from gs_quant.models.risk_model import DataAssetsRequest, RiskModelUniverseIdentifierRequest as UniverseIdentifier, FactorType
import datetime as dt
import matplotlib.pyplot as plt
from itertools import groupby, cycle

# Time range
start_date = dt.date(2021, 1, 1)
end_date = dt.date(2022, 5, 2)

# Macro Model
model_id = "QI_US_EQUITY_LT"
model = MacroRiskModel.get(model_id)

# Asset and factors
asset = "DHR UN"
factors = []

universe = [asset]
universe_for_request = DataAssetsRequest(UniverseIdentifier.bbid, universe)

# Get asset factor sensitivity data
factor_sens_df = model.get_universe_sensitivity(start_date=start_date, end_date=end_date,
                                                assets=universe_for_request, get_factors_by_name=True)

# If no factors passed, get the factors with the highest positive/negative sensitivity
if not factors:
    top_drivers = (
        factor_sens_df.loc[(asset, end_date.strftime("%Y-%m-%d")), :].to_frame()
                                                                     .droplevel(1, axis=1)
                                                                     .sort_values(by=asset, ascending=False)
                                                                     .index.values.tolist()
    )
    factors = top_drivers[0:2] + top_drivers[-2:None]

factor_sens_df = factor_sens_df[factors].droplevel(0)

# Get date range where this asset was in a macro regime
r_squared_historical = (
    model.get_r_squared(start_date=start_date, end_date=end_date,
                        assets=DataAssetsRequest(UniverseIdentifier.bbid, [asset])).reset_index()
                                                                                   .rename(columns={"index": "Date"})
)
macro_regime_dates = r_squared_historical.loc[r_squared_historical[asset] > 65].index.tolist()
temp_list = cycle(macro_regime_dates)
next(temp_list)
groups = groupby(macro_regime_dates, key=lambda j: j + 1 == next(temp_list))
macro_regime_date_range = [list(v) + [next(next(groups)[1])] for k, v in groups if k]

fig, ax = plt.subplots(constrained_layout=True)
factor_sens_df.reset_index().rename(columns={"index": "Date"}).plot(ax=ax, x='Date')
ax.set_title(f"Sensitivity of {universe[0]} to Macro Factors on {end_date} Over Time")
ax.set_ylabel("Percentage change in asset price for 1 std shift")
ax.axhline(y=0, linestyle='dashed')
for date_range in macro_regime_date_range:
    ax.axvspan(date_range[0], date_range[-1], color='aliceblue')
plt.show()

# ### Regime Shift over Time
# 
# In a time range, we can pinpoint points in time when macro factors have primarily explained the variance in asset price over a period of time for some assets in our portfolio. 
# In other words, we can observe when R Squared for an asset was above 65% and when it was below 65%. 
# 
# Here are the user inputs:
# * `model_id`: The model we are using.
# * `universe`: A list of assets to get historical R Squared data for.
# * `start_date` and `end_date`: Time range to get R Squared data

# In[ ]:


import datetime as dt
from gs_quant.target.risk_models import RiskModelDataAssetsRequest as DataRequest, RiskModelUniverseIdentifierRequest
from gs_quant.models.risk_model import MacroRiskModel
import matplotlib.pyplot as plt
import numpy as np

start_date = dt.date(2021, 1, 1)
end_date = dt.date(2022, 5, 16)

model_id = "QI_US_EQUITY_LT"
model = MacroRiskModel.get(model_id)

universe = ["DHR UN", "AMZN UN"]
universe_for_request = DataRequest(RiskModelUniverseIdentifierRequest.bbid, universe)

r_squared_df = model.get_r_squared(start_date=start_date, end_date=end_date, assets=universe_for_request)

fig, ax = plt.subplots(constrained_layout=True)
r_squared_df.plot(ax=ax)
ax.set_title(f"Regime Shift from {start_date} to {end_date}")
ax.set_ylabel("RSquared (in %)")
ax.axhline(y=65, linestyle='dashed')
plt.show()

# ### Historical Fair Value Gap
# 
# The fair value gap is the difference between the model value and asset price. Note that a negative fair value gap (model value < asset price) means that the asset is rich, while a model value > asset price means that the asset is cheap.

# In[ ]:


import datetime as dt
from gs_quant.target.risk_models import RiskModelDataAssetsRequest as DataRequest, RiskModelUniverseIdentifierRequest
from gs_quant.models.risk_model import MacroRiskModel, Unit
from itertools import groupby, cycle
import matplotlib.pyplot as plt

start_date = dt.date(2021, 1, 1)
end_date = dt.date(2022, 5, 2)

model_id = "QI_US_EQUITY_LT"
model = MacroRiskModel.get(model_id)

asset_name = "DHR UN"
universe = [asset_name]
universe_for_request = DataRequest(RiskModelUniverseIdentifierRequest.bbid, universe)

fair_value_gap_std_df = model.get_fair_value_gap(start_date=start_date, end_date=end_date, 
                                                 fair_value_gap_unit=Unit.STANDARD_DEVIATION, 
                                                 assets=universe_for_request)

# Get date range where this asset was in a macro regime
r_squared_historical = (
    model.get_r_squared(start_date=start_date, end_date=end_date, 
                        assets=DataAssetsRequest(UniverseIdentifier.bbid, universe)).reset_index()
                                                                                    .rename(columns={"index": "Date"})
)
macro_regime_dates = r_squared_historical.loc[r_squared_historical[asset_name] > 65].index.tolist()
temp_list = cycle(macro_regime_dates)
next(temp_list)
groups = groupby(macro_regime_dates, key=lambda j: j + 1 == next(temp_list))
macro_regime_date_range = [list(v) + [next(next(groups)[1])] for k, v in groups if k]

fig, ax = plt.subplots(figsize=(15, 7), dpi=150, constrained_layout=True)
fair_value_gap_std_df.reset_index().rename(columns={"index": "Date"}).iloc[:, 0:4].plot(ax=ax, x='Date')
ax.set_title(f"Fair Value Gap from {start_date} to {end_date}", fontsize='x-large', fontweight='bold')
ax.set_ylabel("Fair Value Gap (sigma)", fontsize='large')
ax.axhline(y=0, linestyle='dashed')

for date_range in macro_regime_date_range:
    ax.axvspan(date_range[0], date_range[-1], color='aliceblue')
plt.show()



