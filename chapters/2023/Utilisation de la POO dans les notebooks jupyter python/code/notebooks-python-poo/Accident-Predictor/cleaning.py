import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# dict to store probabilities
Probs = dict()
class Bayesian_Probability:
    def __init__(self, d1, d2):
        self.prob_XA = d1
        self.prob_X  = d2
        
acc_by_road = pd.read_csv("accident-datasets/type-of-road/Acc_clf_acco_to_Road_Cond_2014_and_2016.csv")
acc_by_road.columns
acc_by_road = acc_by_road.drop(labels=[c for c in acc_by_road.columns if '2014' in c], axis=1)
acc_by_road.columns
types_of_road = pd.read_csv("accident-datasets/type-of-road/statewise_road_surfce.csv")
types_of_road.columns
types_of_road = types_of_road.rename(columns={
    'States/UTs':'State/ UT', 'Surfaced-Total (A)':'Surfaced',
    'Unsurfaced-Total (B)':'Unsurfaced', 'Total (A+B)':'Total'
})
types_of_road = types_of_road.drop(labels=['Surfaced-W.B.M.','Surfaced-B.T./C.C'], axis=1)
types_of_road.head()
types_of_road['Surfaced'] = types_of_road['Surfaced']/types_of_road['Total']
types_of_road['Unsurfaced'] = types_of_road['Unsurfaced']/types_of_road['Total']
types_of_road = types_of_road.drop(labels='Total', axis=1)
types_of_road.head()
types_map = {'Surfaced':['Pucca road', 'Speed Breakers', 
                         'Sharp Curve', 'Steep Gradient'],
            'Unsurfaced':['Kutcha road', 'Pot Holes', 
                          'Earthern Shoulder Edge Drop', 'Others']}
acc_by_road = acc_by_road.drop(labels=[c for c in acc_by_road.columns
                                       if 'Persons' in c], axis=1)
acc_by_road.columns
acc_by_road = acc_by_road.rename(columns={
    'Pucca road (Normal Road) - Number of Accidents - 2016':'Pucca road',
    'Kutcha road (Normal Road) - Number of Accidents - 2016':'Kutcha road',
    'Pot Holes - Number of Accidents - 2016':'Pot Holes',
    'Speed Breakers - Number of Accidents - 2016':'Speed Breakers',
    'Sharp Curve - Number of Accidents - 2016':'Sharp Curve',
    'Steep Gradient - Number of Accidents - 2016':'Steep Gradient',
    'Earthern Shoulder Edge Drop - Number of Accidents - 2016':'Earthern Shoulder Edge Drop',
    'Others - Number of Accidents - 2016':'Others'})
acc_by_road = acc_by_road.drop(labels='S. No.', axis=1)
acc_by_road.head()
acc_by_road[[c for c in acc_by_road.columns if c != 'State/ UT']] = acc_by_road[[c for c in acc_by_road.columns 
                                                                                 if c != 'State/ UT']].astype(int)
acc_by_road
acc_by_road['Total'] = acc_by_road[[c for c in acc_by_road.columns
                                      if c != 'State/ UT']].sum(axis=1)
acc_by_road.head()
acc_by_road['Surfaced'] = acc_by_road[types_map['Surfaced']].sum(axis=1)/acc_by_road['Total']
acc_by_road['Unsurfaced'] = acc_by_road[types_map['Unsurfaced']].sum(axis=1)/acc_by_road['Total']
acc_by_road = acc_by_road.drop(labels=[i for i in acc_by_road.columns 
                                       if i not in {'Unsurfaced', 'Surfaced', 'State/ UT'}],
                               axis=1)
acc_by_road.head()
d1 = {i[1][0].lower() : {'Surfaced' : i[1][1],
                 'Unsurfaced' : i[1][2]} for i in acc_by_road.iterrows()}
d1
d2 = {i[1][0].lower() : {'Surfaced' : i[1][1],
                 'Unsurfaced' : i[1][2]} for i in types_of_road.iterrows()}
d2
Probs['Type of Road'] = Bayesian_Probability(d1, d2)
acc_by_drinking = pd.read_csv("accident-datasets/alcohol/acc_due_to alcohol.csv")
acc_by_drinking.columns
acc_by_drinking = acc_by_drinking.drop(labels = ['Sl.No','Total Number of Road Accidents',
'Road accidents due to intake of alcohol as percentage of total accidents in the States',
'Road accidents due to intake of alcohol'], axis = 1)
acc_by_drinking.head()
acc_by_drinking = acc_by_drinking.rename(
    columns={'Percentage Share in total number of road accidents':
            'Drunk'})
acc_by_drinking['Drunk'].sum()
acc_by_drinking['Drunk'] /= 100
acc_by_drinking['Not Drunk'] = 1 - acc_by_drinking['Drunk']
acc_by_drinking.head()
alcohol_prod = pd.read_csv("accident-datasets/alcohol/alcohol-production.csv")
alcohol_prod.columns
alcohol_prod = alcohol_prod.drop(labels=['Production of alcohol* (million litres) - 2013-14',
       'Production of alcohol* (million litres) - 2012-13'], axis = 1)
alcohol_prod = alcohol_prod.drop(index = 12)
alcohol_prod.head()
total_alcohol_prod = alcohol_prod['Production of alcohol* (million litres) - 2014-15'].sum()
total_alcohol_prod
alcohol_prod['Production of alcohol* (million litres) - 2014-15'] /= total_alcohol_prod

alcohol_prod
alcohol_prod = alcohol_prod.rename(columns = {
    'Production of alcohol* (million litres) - 2014-15' : 'Drunk'
})

alcohol_prod['Not Drunk'] = 1 - alcohol_prod['Drunk']
alcohol_prod.head()
d1 = {i[1][0].lower():{'Drunk':i[1][1], 'Not Drunk':i[1][2]} 
      for i in acc_by_drinking.iterrows()}
d2 = {i[1][0].lower():{'Drunk':i[1][1], 'Not Drunk':i[1][2]}
      for i in alcohol_prod.iterrows()}
d1, d2
Probs['Drinking'] = Bayesian_Probability(d1, d2)
acc_by_location = pd.read_csv('accident-datasets/type-of-location/Acc_classified_according_to_Type_of_Location_2014_and_2016.csv')
acc_by_location.head()
acc_by_location = acc_by_location.drop(labels= [c for c in acc_by_location.columns
                                                if '2016' in c or 'Persons' in c]
                                       + ['S. No.'], axis = 1)
acc_by_location.head()
locations = pd.read_csv('accident-datasets/type-of-location/locations.csv', header=None)
locations
locations[1] = locations[1]/locations[1].sum()
locations.head()
acc_by_location = acc_by_location.rename(columns = {'Near School or College - Total Acc. - 2014' : 'school/college',
       'Near or inside a village - Total Acc. - 2014' : 'village',
       'Near a Factory/Industrial area - Total Acc. - 2014' : 'industry/factory' ,
       'Near a religious place - Total Acc. - 2014' : 'religious place',
       'Near a recreation place/cinema - Total Acc. - 2014' : 'recreational place/ cinema',
       'In bazaar - Total Acc. - 2014' : 'market',
       'Near office complex - Total Acc. - 2014' : 'office',
       'Near hospital - Total Acc. - 2014' : 'hospital',
       'Residential area - Total Acc. - 2014': 'residential area',
        'Open area - Total Acc. - 2014' : 'open area',
       'Near bus stop - Total Acc. - 2014': 'bus-stop',
       'Near petrol pump - Total Acc. - 2014' : 'petrol pump',
       'At pedestrian crossing - Total Acc. - 2014' : 'pedestrian crossing',
       'Affected by encroachments - Total Acc. - 2014' : 'enchroachment',
       'Narrow Bridge or culverts - Total Acc. - 2014' : 'narrow-bridge & curves'})
acc_by_location.columns
acc_by_location['Total'] = acc_by_location[[c for c in acc_by_location.columns 
                                     if 'State/ UT' not in c]].sum(axis = 1)
acc_by_location
for c in list(locations[0]):
    acc_by_location[c] = acc_by_location[c]/acc_by_location['Total']
acc_by_location.head()
acc_by_location = acc_by_location.drop(['Total'], axis=1)
d1 = {i[1][0].lower():{ acc_by_location.columns[j]:i[1][j]
                       for j in range(1, acc_by_location.shape[1])}
     for i in acc_by_location.iterrows()}
d2 = {i[1][0].lower():i[1][1] for i in locations.iterrows()}
d1, d2
Probs['Type of Location'] = Bayesian_Probability(d1, d2)
acc_by_license = pd.read_csv("./accident-datasets/type-of-license/Acc_clfacco_to_Type_of_Licence_2016.csv")
acc_by_license.head()
acc_by_license['Total'] = acc_by_license[['Regular', 'Learner', 'Non-Licenced']].sum(axis=1)
acc_by_license
for c in ['Regular', 'Learner', 'Non-Licenced']:
    acc_by_license[c] /= acc_by_license['Total']
acc_by_license.head()
license_types = pd.read_csv("./accident-datasets/type-of-license/prob.csv", header=None)
license_types
d1 = {i[1][1].lower():{'regular':i[1][2],
                       'learner':i[1][3],
                       'non-licensed':i[1][4]} for i in acc_by_license.iterrows()}
d2 = dict(zip(license_types[0], license_types[1]))
d1, d2
Probs['Type of Licence'] = Bayesian_Probability(d1, d2)
acc_by_junc = pd.read_csv("./accident-datasets/type-of-junction/Road_Accidents_2017-Tables_3.5.csv")
acc_by_junc
acc_by_junc = acc_by_junc.drop([5], axis=0)
acc_by_junc
junction = pd.read_csv("./accident-datasets/type-of-junction/junction.csv", header=None)
junction
d1 = {i[1][0].lower() : i[1][2]/100. for i in acc_by_junc.iterrows()}
d2 = {i[1][0].lower() : i[1][1] for i in junction.iterrows()}
d1, d2
Probs['Type of Junction'] = Bayesian_Probability(d1, d2)
acc_by_veh = pd.read_csv("./accident-datasets/type-of-vehicle/roadac2012_AnnexureXIV.csv")
acc_by_veh.head()
acc_by_veh = acc_by_veh.drop([c for c in acc_by_veh.columns
                              if set(c.split()) & {'Fatal', 'Killed', 'Injured', 'Unnamed:'}],
                             axis=1)
acc_by_veh.head()
s = len("Number of Total Road Accidents of ")
s
acc_by_veh = acc_by_veh.rename(columns={
    c:c[s:].lower() for c in acc_by_veh.columns if 'State' not in c
})
acc_by_veh
acc_by_veh['Total'] = acc_by_veh[[c for c in acc_by_veh.columns
                                  if 'State' not in c]].sum(axis=1)
for c in acc_by_veh.columns:
    if 'State' in c:
        continue
    acc_by_veh[c] /= acc_by_veh['Total']
acc_by_veh = acc_by_veh.drop(['Total'], axis=1).fillna(0)
acc_by_veh.head()
vehicles = pd.read_csv("./accident-datasets/type-of-vehicle/counts_of_types_of_vehicles2011.csv")
vehicles.head()
vehicles = vehicles.fillna(0)
vehicles["trucks, tempos, mavs, tractors"] =\
vehicles['TRANSPORT-Multi-axled/ Articulated Vehicles/ Trucks and Lorries'] \
+ vehicles['TRANSPORT-Light Motor Vehicle (Goods)'] \
+ vehicles['NON-TRANSPORT-Tractors'] \
vehicles['cars, jeeps, taxis'] = vehicles["TRANSPORT-Taxis"] \
+ vehicles['NON-TRANSPORT-Cars'] \
+ vehicles['NON-TRANSPORT-Jeeps'] \
vehicles['auto-rickshaws'] = vehicles['TRANSPORT-Light Motor Vehicle (Passenger)']
vehicles['two-wheelers'] = vehicles['NON-TRANSPORT-Two Wheelers']
vehicles['buses'] = vehicles['TRANSPORT-Buses'] + vehicles['NON-TRANSPORT-Omni Buses']
vehicles['other motor vehicles'] = vehicles['NON-TRANSPORT-Others']
vehicles['other vehicles/objects'] = vehicles['NON-TRANSPORT-Trailers']
vehicles['Total'] = vehicles[[c for c in acc_by_veh.columns if 'State' not in c]].sum(axis=1)
for c in acc_by_veh.columns:
    if 'State' not in c:
        vehicles[c] /= vehicles['Total']
vehicles = vehicles.drop([c for c in vehicles.columns
                         if c != "STATES/UTs" and c not in acc_by_veh.columns], axis=1)
vehicles.head()
d1 = {i[1][0].lower():{ acc_by_veh.columns[j]:i[1][j] for j in range(1, acc_by_veh.shape[1]) }
     for i in acc_by_veh.iterrows()}
d2 = {i[1][0].lower():{ vehicles.columns[j]:i[1][j] for j in range(1, vehicles.shape[1]) }
     for i in vehicles.iterrows()}
d1, d2
Probs['Type of Vehicle'] = Bayesian_Probability(d1, d2)
priors = pd.read_csv("./accident-datasets/2014_an5.csv")
priors.head()
priors = priors.drop(columns=[c for c in priors.columns
                             if c != "Total Number of Persons Injured in Road Accidents per 10,000 Vehicles - 2012"
                             and c != "States/UTs"])
priors.head()
priors["Total Number of Persons Injured in Road Accidents per 10,000 Vehicles - 2012"] /= 10000
priors.head()
d = {i[1][0].lower() : i[1][1] for i in priors.iterrows()}
d
Probs['Priors'] = Bayesian_Probability(d, None)
Probs
# save dict as pickle
from pickle import dump, load
with open("./cleaned/Probs.dat", "wb") as fh:
    dump(Probs, fh)
# load pickle
with open("./cleaned/Probs.dat", "rb") as fh:
    Probs = load(fh)
Probs

