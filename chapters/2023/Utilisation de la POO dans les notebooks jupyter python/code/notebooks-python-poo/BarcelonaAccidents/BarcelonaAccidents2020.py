import functions_barcelona as fb
#from functions_barcelona import posant_accents,utmToLatLng,\
      #              ped_to_angles,setmana_a_angles,mes_a_angles,cause_to_angles
import datetime
import math
import pickle
import pandas as pd
import requests
import json
import os
import numpy as np
from datetime import date, datetime, timedelta
#from functions_barcelona import getting_daily_weather,getting_next_day
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
import webbrowser
import autoreload
%load_ext autoreload
%autoreload 2



r = requests.get('https://opendata-ajuntament.barcelona.cat/data/api/3/action/package_search?rows=1000')
response= r.json()


def concatenating_dataframes(filter1):
    
    files=response['result']['results']
    for num, file in enumerate(files):
        
        if ('accidents-' +filter1+'gu') in file['name']:
            print(file['name'])
            filter_list=[]
            
            for fitxer in file['resources']:

                if fitxer['format']=='CSV':

                    print(fitxer['name'],fitxer['format'])
                    #print(fitxer['url'])
                    try:
                        filter_list.append(pd.read_csv(fitxer['url']))
                    except:
                        try:
                            filter_list.append(pd.read_csv(fitxer['url'],encoding='ISO-8859-15'))
                        except:

                            filter_list.append(pd.read_csv(fitxer['url'],sep=';',encoding='ISO-8859-1'))

            column_names=filter_list[0].columns
            data=pd.DataFrame()
            for df in filter_list:
                df=pd.DataFrame.from_records(df.values)
                data=pd.concat([data,df])
                #print(data.shape)
            data.columns=column_names
            data=data.reset_index()
    try:
        return data
    except:
        print('NO FILE WITH THAT FILTER')
    print('CONCATENATING DONE')
#causes_df= concatenating_dataframes('causes-')

noms=['causes-', 'persones-','tipus-', 'vehicles-','']
dict_noms={}
for nom in noms:
    dict_noms[nom[:-1]]=concatenating_dataframes(nom)

pickle.dump( dict_noms, open( "../data/dataframes_dict.pkl", "wb" ) )
#favorite_color = pickle.load( open( "dataframes_dict.pkl", "rb" ) )
dict_noms = pickle.load( open( "../data/dataframes_dict.pkl", "rb" ) )
acc= dict_noms[''].copy()

#acc.columns
index_to_drop=acc[acc.isnull().sum(axis=1)>10].index
acc=acc.drop(index_to_drop)
acc['Nom_carrer']=acc['Nom_carrer'].fillna('Casetes')
for column in acc:
    if 'Coor' in column:
        print(column)
        acc[column]=[str(coor).replace(',','.') for coor in acc[column]]
        acc[column]= acc[column].astype(float)
for column in acc:
    acc[column]=acc[column].apply(fb.posant_accents)
old_column_names=acc.columns
acc=acc.drop(['Num_postal_caption','Longitud','Latitud','index','Codi_barri', 'Dia_setmana','Mes_any',
             'Descripcio_torn', 'Descripcio_tipus_dia'],axis=1)
print('Number of nulls:',acc.isnull().sum().sum())

acc['Codi_districte']=[x if x!='Desconegut' else -1 for x in acc['Codi_districte']]
acc['Codi_carrer']=acc['Codi_carrer'].astype(int)
#acc['Codi_barri']=[str(x)[-2:].replace('-','') for x in acc.Codi_barri]
tuple_features=[('Nom_carrer','Codi_carrer'),('Nom_districte','Codi_districte')]
for tup in tuple_features:
    if acc[tup[0]].nunique()==acc[tup[1]].nunique():
        acc=acc.drop(tup[1],axis=1)
        print(f"I dropped {tup[1]}")

print(acc.shape)

acc.columns=['num_incident', 'district', 'neighborhood', 'street_code',
       'street_name', 'weekday', 'year', 'month', 'day', 'hour', 'ped_cause',
       'num_deaths', 'num_minorly_injured', 'num_severly_injured',
       'num_victims', 'num_vehicles', 'utm_y',
       'utm_x']
##Fixing utm_x and utm_y that are mixed in some cases. Replacing nulls(-1) with the mean
acc['utm_x']=[x[0] if len(str(x[0]).split('.')[0])==7 else x[1]  if len(str(x[1]).split('.')[0])==7 else round(acc.utm_x.mean(),2) for x in zip(acc.utm_x,acc.utm_y)]
acc['utm_y']=[x[1] if len(str(x[1]).split('.')[0])==6 else x[0]  if len(str(x[0]).split('.')[0])==6 else round(acc.utm_y.mean(),2) for x in zip(acc.utm_x,acc.utm_y)]


##Translating to English

acc['ped_cause']=acc['ped_cause'].apply(fb.ped_to_angles)
acc['weekday']=acc['weekday'].apply(fb.setmana_a_angles)
acc['month']=acc['month'].apply(fb.mes_a_angles)
acc['num_incident']=[x.strip() for x in acc['num_incident']]
for col in ['year','day','hour','num_deaths','num_minorly_injured','num_severly_injured','num_victims','num_vehicles']:
    acc[col]=acc[col].astype(int)
##eliminatong duplictaes that have 2 ped cause: I am losing the second one.
index_to_drop= acc[acc['num_incident'].duplicated()].index
acc=acc.drop(index_to_drop).reset_index()
acc.to_csv('../data/accidents_only2020.csv')
print('Accidents: ', acc.shape)

##adding causes
causes= dict_noms['causes'].reset_index().copy()


columns_to_keep=['Descripcio_causa_mediata', 'Numero_expedient']

causes=causes[columns_to_keep]
causes.columns=['cause','num_incident']
causes['num_incident']=[x.strip() for x in causes['num_incident']]
causes['cause']=causes.cause.apply(fb.posant_accents).apply(fb.cause_to_angles)
print(causes.shape)
causes=causes.drop_duplicates('num_incident')


causes['cause']=causes['cause'].fillna('unknown')
causes.to_csv('../data/causes2020.csv')
total= pd.merge(acc,causes, how='left',on='num_incident')

total['cause']=total.cause.fillna('unknown')

print('causes: ', causes.shape, 'Total: ',total.shape)



##people

people= dict_noms['persones'].reset_index().copy()

columns_to_add=['Numero_Expedient','Desc_Tipus_vehicle_implicat', 'Descripcio_sexe', 'Edat',
                'Descripcio_tipus_persona', 'Descripcio_Lloc_atropellament_vianat',
                'Descripcio_Motiu_desplaçament_vianant',
                'Descripcio_Motiu_desplaçament_conductor', 'Descripcio_victimitzacio',]
people.columns=[fb.posant_accents(col) for col in people.columns]
anual_dict={}
for any_ in sorted(people['NK_ Any'].unique()):
    if any_ in [2010,2011,2012,2013]:
        df=people[people['NK_ Any']==any_].copy()
        df.rename(columns={'Edat':'Descripcio_tipus_persona_','Descripcio_tipus_persona':'Edat_',\
        'Descripcio_Motiu_desplaçament_conductor':'Coordenada_UTM_Y_',
        'Descripcio_Motiu_desplaçament_vianant': 'Coordenada_UTM_X_',
        'Descripcio_Lloc_atropellament_vianat':'Descripcio_victimitzacio_' },inplace=True)
        df=df.drop(df.columns[-5:],axis=1)
        #columnes=[]
        df.columns=[col if col[-1]!='_' else col[:-1] for col in df.columns]
        columnes=[col for col in df.columns if col in columns_to_add]
        df=df[columnes].copy()
        anual_dict[any_]=df
    elif any_ in [2014,2015]:
        df=people[people['NK_ Any']==any_].copy()
        df.rename(columns={'Descripcio_Motiu_desplaçament_vianant':'Coordenada_UTM_Y_','Descripcio_Lloc_atropellament_vianat':'Coordenada_UTM_X_',\
        'Descripcio_tipus_persona':'Descripcio_victimitzacio_',
        'Descripcio_sexe': 'Descripcio_tipus_persona_',
        'Desc_Tipus_vehicle_implicat':'Descripcio_sexe_',
         'Descripcio_causa_vianant':  'Desc_Tipus_vehicle_implicat_'   },inplace=True)
        df=df.drop(df.columns[-6:],axis=1)
        df.columns=[col if col[-1]!='_' else col[:-1] for col in df.columns]
        columnes=[col for col in df.columns if col in columns_to_add]
        df=df[columnes].copy()
        anual_dict[any_]=df
    elif any_ in [2016,2017,2018]:
        df=people[people['NK_ Any']==any_].copy()
        df.rename(columns={'Descripcio_Motiu_desplaçament_vianant':'Descripcio_victimitzacio_','Descripcio_Lloc_atropellament_vianat':'Descripcio_situacio',
        },inplace=True)
        df=df.drop(df.columns[-6:],axis=1)
        df.columns=[col if col[-1]!='_' else col[:-1] for col in df.columns]
        columnes=[col for col in df.columns if col in columns_to_add]
        df=df[columnes].copy()
        anual_dict[any_]=df
    elif any_ in [2019,2020]:
        df=people[people['NK_ Any']==any_].copy()
        columnes=[col for col in df.columns if col in columns_to_add]
        df=df[columnes].copy()
        anual_dict[any_]=df

mapping_columns={'Numero_Expedient':'num_incident',
             'Desc_Tipus_vehicle_implicat':'vehicle',
             'Descripcio_sexe':'gender',
             'Edat':'age',
            'Descripcio_tipus_persona':'people_role',
             'Descripcio_Lloc_atropellament_vianat':'run_over_location',
             'Descripcio_victimitzacio': 'level_injuries',
              'Descripcio_Motiu_desplaçament_vianant':'peds_activity',
            'Descripcio_Motiu_desplaçament_conductor':'drivers_activity'}
people=pd.DataFrame()
for key in anual_dict.keys():
    #print(key, anual_dict[key].shape)
    people= pd.concat([people,anual_dict[key].rename(columns=mapping_columns)])
    #print(people.shape)
    ##too many nulls.
for col in people:
    if people[col].isnull().sum()>10000:
        people=people.drop(col,axis=1)
people=people.drop('level_injuries',axis=1)
people['num_incident']=[x.strip() for x in people['num_incident']]
people['age']=people.age.astype(str)
people['age']=[edat if edat!='Desconegut' else '-1' for edat in people.age]
people['vehicle']=people.vehicle.map(fb.map_vehicles)
people['gender']=people.gender.map({'Home':'M','Dona':"F",'Desconegut':'unknown'})
people['people_role']=people.people_role.map({'Conductor':'driver','Passatger':'pass','Vianant':'ped','Desconegut':'unknown'})
#compressed_people= people.groupby('num_incident').agg(lambda x : ','.join(x)).reset_index()
people.to_csv('../data/people2020.csv')
##Fixing people
misc_vehicles=people.vehicle.value_counts()[people.vehicle.value_counts()<1000].index
people['vehicles']=[veh if veh not in misc_vehicles else 'misc_vehicle' for veh in people.vehicle]

if 'vehicle' in list(people.columns):
    people=people.drop('vehicle',axis=1)
people=people.applymap(lambda x: np.nan if x in ['Unknown','-1'] else x)
people['gender']=people.gender.map({'M':0,'F':1})
people['age']=people['age'].astype(float)
people=people.set_index('num_incident')
people2=pd.get_dummies(people)


scaler = MinMaxScaler()
people2 = pd.DataFrame(scaler.fit_transform(people2), columns = people2.columns)


imputer = KNNImputer(n_neighbors=5)
people2 = pd.DataFrame(imputer.fit_transform(people2),columns = people2.columns)


people2=pd.DataFrame(scaler.inverse_transform(people2), columns = people2.columns)
people2['gender_100/1']=[100 if gen<0.5 else 1 for gen in people2.gender]
people2['age_driver']=people2.age*people2.people_role_driver
people2['gender_driver']=people2['gender_100/1']*people2.people_role_driver
if 'gender' in list(people2.columns):
    people2=people2.drop('gender',axis=1)
people2=people2.set_index(people.index)
people2=people2.reset_index().groupby('num_incident').sum()
people2['age_driver']=[x[0]/x[1] if x[1]!=0 else 0 for x in zip(people2['age_driver'],people2['people_role_driver'])]
people2['gender_driver_male']=[int(str(gen)[0]) if len(str(gen))==3 else 0 for gen in people2.gender_driver.astype(int)]
people2['gender_driver_female']=[int(str(gen)[2]) if len(str(gen))==3 else gen for gen in people2.gender_driver.astype(int)]
people2=people2.drop(['age','gender_100/1','gender_driver'],axis=1)

total= pd.merge(total,people2, how='left',on='num_incident')
total['age_driver']=total.age_driver.fillna(total.age_driver.mean())
total=total.fillna(0)
#total=total.fillna(-1)
total3=total.copy()

total.sample(5)
print('people: ', people.shape, 'Total: ',total.shape)

##TYPE
tipus=dict_noms['tipus'].copy()
type_accident_map={'Atropellament': 'run_over',
         'Col.lisio lateral': 'lateral_collision',
        'Xoc contra element estatic': 'crash_into_stationary',
      'Abast': 'rear-end_collision',
       'Col.lisio frontal':'frontal_collision',
      'Col.lisio fronto-lateral':'frontal-lateral_collision',
      'Caiguda (dues rodes)':'fall--motorcycle',
      'Abast multiple':'multiple_rear-end_collision',
      'Caiguda interior vehicle':'fall_inside_vehicle',
      'Altres':'Other_types',
      'Bolcada (mes de dues rodes)':'overturning',
      'Desconegut':'unknown',
      'Sortida de via amb xoc o col.lisio':'run-off_with_crash_or_collision',
      'Encalç':'rear-end_collision',
      'Sortida de via amb bolcada':'run-off_with_overturning',
      'Xoc amb animal a la calçada':'crash_into_animal_on_road',
      'Resta sortides de via':'run-off_not_included_previously'}

tipus['Descripcio_tipus_accident']=tipus['Descripcio_tipus_accident'].apply(fb.posant_accents).map(type_accident_map)
tipus=tipus[['Numero_expedient','Descripcio_tipus_accident',]].copy()
tipus.columns=['num_incident','accident_type']
tipus['num_incident']=[x.strip() for x in tipus.num_incident]
tipus=tipus.dropna()
misc_type=list(tipus.accident_type.value_counts()[tipus.accident_type.value_counts()<350].index)
misc_type.append('Other_types')
tipus['accident_type']=["misc_type" if x in misc_type else 'collision' if 'collision' in x else 'crash' if 'crash' in x else x for x in tipus.accident_type]

tipus=tipus.groupby('num_incident')['accident_type'].agg(lambda x: ','.join(x)).reset_index()
type_list=tipus.accident_type.value_counts().index
tipus['accident_type']=[fb.organizing_types(x,type_list) for x in  tipus.accident_type]
#tipus2=tipus.copy()
total= pd.merge(total,tipus, how='left',on='num_incident')
total.fillna('misc_type',inplace=True)
total4=total.copy()
tipus.to_csv('../data/types2020.csv')

print('type: ', tipus.shape, 'Total: ',total.shape)

##VEHICLE***Model no surt a tots els anys

vehicles= dict_noms['vehicles'].copy()
columns_to_add=['Numero_expedient', 'Descripcio_model', 'Descripcio_marca',
       'Descripcio_color', 'Descripcio_carnet', 'Antiguitat_carnet' ]

anual_dict={}

for any_ in sorted(vehicles['NK_Any'].unique()):
    if any_ in [2010,2011,2013,2014,2015,2016,2017]:
        df=vehicles[vehicles['NK_Any']==any_].copy()
        df.rename(columns={'Descripcio_tipus_vehicle':'Descripcio_model_','Descripcio_model':'Descripcio_marca_',\
        'Descripcio_marca':'Descripcio_color_',
        'Descripcio_color': 'Descripcio_carnet_',
        'Descripcio_carnet':'Antiguitat_carnet_' },inplace=True)
        df=df.drop(df.columns[-5:],axis=1)
        df.columns=[col if col[-1]!='_' else col[:-1] for col in df.columns]
        columnes=[col for col in df.columns if col in columns_to_add]
        df=df[columnes].copy()
        anual_dict[any_]=df
    elif any_ in [2018,2019,2020]:
        df=vehicles[vehicles['NK_Any']==any_].copy()        
        columnes=[col for col in df.columns if col in columns_to_add]
        df=df[columnes].copy()
        anual_dict[any_]=df

vehicles=pd.DataFrame()
for key in anual_dict.keys():
    vehicles=pd.concat([vehicles,anual_dict[key]],)
    

vehicles.columns=['num_incident', 'vehicle_model', 'vehicle_brand',
       'vehicle_color', 'type_license', 'seniority_license']

vehicles=vehicles.dropna()
vehicles['num_incident']=[x.strip() for x in vehicles.num_incident]
vehicles2=vehicles.copy()
vehicles=vehicles.groupby('num_incident').agg(lambda x: ','.join(x)).reset_index()
vehicles.to_csv('../data/vehicles2020.csv')
total= pd.merge(total,vehicles, how='left',on='num_incident')
##I will imput the most often
total=total.apply(lambda x: x.fillna(x.value_counts().index[0]))

total.to_csv('../data/accidents2020.csv')
#total.fillna(-1,inplace=True)
print('vehicles: ', vehicles.shape, 'Total: ',total.shape)
print('DONE')
url = "https://www.youtube.com/watch?v=Udt-9J8nzGE"
webbrowser.open(url,new=1)
##REPEAT THE PROCESS FOR EACH YEAR----I have to limit the number of calls therefore I have to do it by year

# import datetime
# start_date = datetime.date(2020, 1, 1)
# end_date = datetime.date(2020, 12, 31)
# delta = datetime.timedelta(days=1)
# df_total = []
# while start_date <= end_date:
#     print(start_date)
#     df= getting_daily_weather(str(start_date))
#     df_total.append(df)
#     start_date += delta
# pd.concat(df_total).to_csv('./data/weather2020.csv')
weather_df=pd.DataFrame()

##Combining every year

for file in os.listdir('../data/'):
    if 'weather' in file:
        #print(file)#,pd.read_csv('../data/'+file).shape,pd.read_csv('../data/'+file).columns)
        weather_df=pd.concat([weather_df,pd.read_csv('../data/'+file)])
weather_df=weather_df.sort_values('time')
weather_df=weather_df.reset_index(drop=True)
for col in weather_df:
    if 'Unnamed' in col:
        weather_df=weather_df.drop(col,axis=1)
weather_df['time']=pd.to_datetime(weather_df.time)
weather_df=weather_df.drop_duplicates('time')
weather_df=weather_df.loc[:,weather_df.isnull().sum()<25]
calendar_df=pd.DataFrame(pd.date_range("2010-01-01", "2021-01-01", freq="H"),columns=['calendar_date'])[:-1]
weather_df=pd.concat([calendar_df,weather_df],axis=1)
weather_df=weather_df.fillna(method='ffill')
weather_df.columns=['weather_'+col if 'calendar' not in col else col for col in weather_df.columns]
weather_df['weather_icon']=['partly-cloudy' if col.startswith('partly-cloudy') else 'clear' if col.startswith('clear') else 'rare' if\
                            col in ['fog','snow','sleet','wind'] else col for col in weather_df.weather_icon]
weather_df['weather_deadly_temperature']=[1 if temp>55 and temp<70 else 0 for temp in weather_df.weather_temperature]
weather_only=weather_df.copy()


#MERGING WEATHER AND TOTAL

total['month']=total['month'].apply(fb.mes_english_number)
date_columns = ['month', 'day', 'year','hour']
date = total[date_columns]
dates = []
datetimes=[]
for i in date.itertuples():
    dates.append(i[1]+'/' +str(int(i[2]))+'/'+str(int(i[3])))
    datetimes.append(i[1]+'/' +str(int(i[2]))+'/'+str(int(i[3]))+' '+str(int(i[4]))+':00:00')

total['dates']=dates
total['datetimes']=datetimes
total['dates']=pd.to_datetime(total.dates)
total['datetimes']=pd.to_datetime(total.datetimes)
total=pd.merge(total,weather_df,how='left', right_on='calendar_date',left_on='datetimes')
total=total.drop(columns=['weather_time','calendar_date','weather_apparentTemperature'])
total.to_csv('../data/weather_accidents_2010_20.csv')
webbrowser.open(url,new=1)


