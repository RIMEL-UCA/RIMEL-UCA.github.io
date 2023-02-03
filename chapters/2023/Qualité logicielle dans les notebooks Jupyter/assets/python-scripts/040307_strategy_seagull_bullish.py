#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.common import AggregationLevel
from gs_quant.instrument import IRSwaption
from gs_quant.backtests.triggers import PeriodicTrigger, PeriodicTriggerRequirements
from gs_quant.backtests.actions import AddTradeAction
from gs_quant.backtests.strategy import Strategy
from gs_quant.backtests.generic_engine import GenericEngine
from gs_quant.markets.portfolio import Portfolio
from gs_quant.markets import HistoricalPricingContext
from gs_quant.risk import IRDelta, Price, IRFwdRate, IRDailyImpliedVol, DollarPrice
from datetime import date
from gs_quant.datetime import business_day_offset

# In[ ]:


# Initialize session
from gs_quant.session import GsSession
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics',)) 

# In[ ]:


# Seagull params
start = date(2020, 1, 1)
end = business_day_offset(date.today(), -1, roll='preceding')

# Add trades at what frequency
freq_int = 1
ramp_up_freq = 'w'
freq = str(freq_int) + ramp_up_freq
ramp_up_mult = 5
ramp_up_period = ramp_up_mult * freq_int - 1
holding_period = str(ramp_up_period) + ramp_up_freq
strategy_notional = 100e6

print(freq,holding_period)

# ## Buy call spread, sell out-of-the-money put

# In[ ]:


# Portfolio def

ccy = 'USD'
expiry = '1y'
tail = '10y'
rate_option = 'USD-LIBOR-BBA'
pay_rec = ['Receiver', 'Receive', 'Pay']
strikes = ['A-50', 'A-25', 'A+50']
direction = [-1, 1, -1]
notional_inception = round(strategy_notional/(ramp_up_period+1), 0)

port = []
for i in range(len(pay_rec)):
    opt = IRSwaption(pay_or_receive=pay_rec[i],
                     termination_date=tail,
                     notional_currency=ccy,
                     expiration_date=expiry,
                     notional_amount=direction[i]*notional_inception,
                     strike = strikes[i],
                     floating_rate_option=rate_option,
                     name='swaption'+str(i))
    port.append(opt)

csa_string = ccy + '-1'    
    
seagull = Portfolio(port, name='Seagull')
seagull.to_frame().transpose()

# In[ ]:


# Define deltas
myParallelDelta = IRDelta(currency='local', aggregation_level=AggregationLevel.Type)

# Define backtest
trade_action = AddTradeAction(seagull, holding_period)
trigger = PeriodicTrigger(PeriodicTriggerRequirements(start_date=start, end_date=end, frequency=freq), [trade_action])
strategy = Strategy(None, trigger)

# Select an engine
GE = GenericEngine()
backtest = GE.run_backtest(strategy, start=start, end=end,frequency=freq, risks=[Price, myParallelDelta], 
                           show_progress=True, csa_term=csa_string)

# In[ ]:


backtest_summary = backtest.result_summary
backtest_summary

# In[ ]:


backtest_dates_reduced = backtest_summary.index

with HistoricalPricingContext(dates = backtest_dates_reduced, show_progress=True):
    ir_rates = seagull[0].calc([IRFwdRate, IRDailyImpliedVol])
    entry = seagull.calc([DollarPrice])


premium = entry.result().to_frame()
raw_data = ir_rates.result().to_frame()
raw_data[IRFwdRate] *= 1e4
raw_data['entry_premium'] = (premium.iloc[:,0]+premium.iloc[:,1]+premium.iloc[:,2])
raw_data.describe()

# In[ ]:


(backtest_summary['Cumulative Cash']/1e6).plot(title='Rolling Seagull vs Delta Proxy', legend=True, label='Strategy', figsize=(20, 10))
(backtest_summary.iloc[5, 1] * (raw_data[IRFwdRate]-raw_data[IRFwdRate][0])/1e6).plot(legend=True, 
                            label=f'Delta Proxy ({round(backtest_summary.iloc[5, 1]/1e3, 0)}k per bp)')
