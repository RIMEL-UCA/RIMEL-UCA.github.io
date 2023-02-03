import pandas as pd
class Bayesian_Probability:
    def __init__(self, d1, d2):
        self.prob_XA = d1
        self.prob_X  = d2
from pickle import load
with open("./cleaned/Probs.dat", "rb") as fh:
    Probs = load(fh)
Probs
from collections import defaultdict
state_map = defaultdict(list)   # to store alternate names
state1 = sorted(Probs['Type of Road'].prob_XA.keys())
state2 = sorted(Probs['Type of Road'].prob_X.keys())
print(state1, state2, sep='\n')
state1 = sorted(Probs['Drinking'].prob_XA.keys())
state2 = sorted(Probs['Drinking'].prob_X.keys())
print(state1, state2, sep='\n')
state1 = sorted(Probs['Type of Licence'].prob_XA.keys())
state2 = sorted(Probs['Type of Licence'].prob_X.keys())
print(state1, state2, sep='\n')
state1 = sorted(Probs['Type of Location'].prob_XA.keys())
state2 = sorted(Probs['Type of Location'].prob_X.keys())
print(state1, state2, sep='\n')
state1 = sorted(Probs['Type of Vehicle'].prob_XA.keys())
state2 = sorted(Probs['Type of Vehicle'].prob_X.keys())
print(state1, state2, sep='\n')
default = ['total', 'others', 'all india']
state_map['andaman and nicobar islands'].extend(['andaman and nicobar islands',
                                                 'a & n islands', 'andaman and nicobar',
                                                 'andaman & nicobar', 'a & n',
                                                 'andaman & nicobar islands'])
state_map['andhra pradesh'].extend(['andhra pradesh'])
state_map['arunachal pradesh'].extend(['arunachal pradesh'])
state_map['assam'].extend(['assam' ])
state_map['bihar'].extend(['bihar' ])
state_map['chandigarh'].extend(['chandigarh', 'punjab', 'haryana' ])
state_map['chhattisgarh'].extend(['chhattisgarh' ])
state_map['dadra & nagar haveli'].extend(['dadra & nagar haveli', 'd & n haveli',
                                          'dadra and nagar haveli' ])
state_map['daman & diu'].extend(['daman & diu', 'daman and diu' ])
state_map['delhi'].extend(['delhi', 'new delhi', 'nct of delhi' ])
state_map['goa'].extend(['goa' ])
state_map['gujarat'].extend(['gujarat' ])
state_map['haryana'].extend(['haryana' ])
state_map['himachal pradesh'].extend(['himachal pradesh', 'himachal' ])
state_map['jammu & kashmir'].extend(['jammu & kashmir', 'jammu and kashmir',
                                     'j & k' ])
state_map['jharkhand'].extend(['jharkhand' ])
state_map['karnataka'].extend(['karnataka', 'karnatak' ])
state_map['kerala'].extend(['kerala', 'keral' ])
state_map['lakshadweep'].extend(['lakshadweep' ])
state_map['madhya pradesh'].extend(['madhya pradesh', 'm.p.', 'mp' ])
state_map['maharashtra'].extend(['maharashtra' ])
state_map['manipur'].extend(['manipur' ])
state_map['meghalaya'].extend(['meghalaya' ])
state_map['mizoram'].extend(['mizoram' ])
state_map['nagaland'].extend(['nagaland' ])
state_map['orissa'].extend(['orissa', 'odisha', 'odhisha' ])
state_map['puducherry'].extend(['puducherry', 'pudducherry', 'pondicherri',
                                'tamil nadu & puducherry' ])
state_map['punjab'].extend(['punjab' ])
state_map['rajasthan'].extend(['rajasthan' ])
state_map['sikkim'].extend(['sikkim' ])
state_map['tamil nadu'].extend(['tamil nadu', 't. nadu', 't nadu', 'tn',
                                'tamil nadu & puducherry'])
state_map['telangana'].extend(['telangana', 'andhra pradesh' ])
state_map['tripura'].extend(['tripura' ])
state_map['uttar pradesh'].extend(['uttar pradesh', 'u.p.', 'up' ])
state_map['uttarakhand'].extend(['uttarakhand' ])
state_map['west bengal'].extend(['west bengal', 'w. bengal', 'w bengal', 'bengal',
                                 'west bangal', 'w bangal', 'w. bangal', 'bangal'])
for k in state_map:
    state_map[k].extend(default)
state_map = dict(state_map)
state_map
from pickle import dump
with open("./cleaned/state-map.dat", 'wb') as fh:
    dump(state_map, fh)
from decimal import Decimal
class NaiveBayes:
    def __init__(self, Probs, state_map):
        self.Probs = Probs
        self.state_map = state_map
    
    @staticmethod
    def get_state_prob(d, state, state_map):
        for s in state_map[state]:
            if s in d:
                return d[s]
    
    def road_prob(self, state, type_of_road):
        pxa = __class__.get_state_prob(self.Probs['Type of Road'].prob_XA, 
                             state, self.state_map)[type_of_road]
        px = __class__.get_state_prob(self.Probs['Type of Road'].prob_X,
                            state, self.state_map)[type_of_road]
        if pxa == 0 or px == 0:
            return 1
        return Decimal(pxa)/Decimal(px)

    def location_prob(self, state, location):
        pxa = __class__.get_state_prob(self.Probs['Type of Location'].prob_XA, 
                             state, self.state_map)[location]
        px = self.Probs['Type of Location'].prob_X[location]
        if pxa == 0 or px == 0:
            return 1
        return Decimal(pxa)/Decimal(px)
        
    def licence_prob(self, state, licence):
        pxa = __class__.get_state_prob(self.Probs['Type of Licence'].prob_XA, 
                             state, self.state_map)[licence]
        px = self.Probs['Type of Licence'].prob_X[licence]
        if pxa == 0 or px == 0:
            return 1
        return Decimal(pxa)/Decimal(px)
    
    def vehicle_prob(self, state, vehicle):
        pxa = __class__.get_state_prob(self.Probs['Type of Vehicle'].prob_XA, 
                             state, self.state_map)[vehicle]
        px = __class__.get_state_prob(self.Probs['Type of Vehicle'].prob_X,
                            state, self.state_map)[vehicle]
        if pxa == 0 or px == 0:
            return 1
        return Decimal(pxa)/Decimal(px)
    
    def alcohol_prob(self, state, drunk_or_not):
        pxa = __class__.get_state_prob(self.Probs['Drinking'].prob_XA, 
                             state, self.state_map)[drunk_or_not]
        px = __class__.get_state_prob(self.Probs['Drinking'].prob_X,
                            state, self.state_map)[drunk_or_not]
        if pxa == 0 or px == 0:
            return 1
        return Decimal(pxa)/Decimal(px)
    
    def junction_prob(self, junction):
        pxa = self.Probs['Type of Junction'].prob_XA[junction]
        px = self.Probs['Type of Junction'].prob_X[junction]
        if pxa == 0 or px == 0:
            return 1
        return Decimal(pxa)/Decimal(px)
    
    def prior_prob(self, state):
        return Decimal(__class__.get_state_prob(self.Probs['Priors'].prob_XA,
                                                state, self.state_map))
    
    def get_probability(self, state, type_of_road, location, 
                        licence, vehicle, drunk=False, junction=None):
        p_road = self.road_prob(state, type_of_road)
        p_loc = self.location_prob(state, location)
        p_lic = self.licence_prob(state, licence)
        p_veh = self.vehicle_prob(state, vehicle)
        p_prior = self.prior_prob(state)
        P = p_road * p_loc * p_lic * p_veh * p_prior
        if drunk:
            P *= self.alcohol_prob(state, 'Drunk')
        if junction is not None:
            P *= self.junction_prob(junction)
        return P
predictor = NaiveBayes(Probs, state_map)
p = predictor.get_probability('delhi', 'Unsurfaced', 'enchroachment', 'regular',
                          'trucks, tempos, mavs, tractors')*100
print("%.2f%%"%p)

