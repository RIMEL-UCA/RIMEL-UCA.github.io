# Import initial dependencies
import os
import pandas as pd
import numpy as np
# Read CSVs into dataframes
stops10 = pd.read_csv("stops_2010.csv", encoding="utf-8", low_memory=False)
stops11 = pd.read_csv("stops_2011.csv", encoding="utf-8", low_memory=False)
stops12 = pd.read_csv("stops_2012.csv", encoding="utf-8", low_memory=False)
stops13 = pd.read_csv("stops_2013.csv", encoding="utf-8", low_memory=False)
stops14 = pd.read_csv("stops_2014.csv", encoding="utf-8", low_memory=False)
stops15 = pd.read_csv("stops_2015.csv", encoding="utf-8", low_memory=False)
# Cut down to desired columns
stops10 = stops10[['stop_date', 'county_name', 'driver_gender',
        'driver_race', 'search_conducted', 
        'contraband_found', 'stop_outcome', 'officer_id']]
stops11 = stops11[['stop_date', 'county_name', 'driver_gender',
        'driver_race', 'search_conducted', 
        'contraband_found', 'stop_outcome', 'officer_id']] 
stops12 = stops12[['stop_date', 'county_name', 'driver_gender',
        'driver_race', 'search_conducted', 
        'contraband_found', 'stop_outcome', 'officer_id']] 
stops13 = stops13[['stop_date', 'county_name', 'driver_gender',
        'driver_race', 'search_conducted', 
        'contraband_found', 'stop_outcome', 'officer_id']] 
stops14 = stops14[['stop_date', 'county_name', 'driver_gender',
        'driver_race', 'search_conducted', 
        'contraband_found', 'stop_outcome', 'officer_id']] 
stops15 = stops15[['stop_date', 'county_name', 'driver_gender',
        'driver_race', 'search_conducted', 
        'contraband_found', 'stop_outcome', 'officer_id']] 
# Clean column names
stops10.columns = ['date', 'county', 'gender', 'race',
       'searched', 'contraband', 'outcome', 'officer_id']
stops11.columns = ['date', 'county', 'gender', 'race',
       'searched', 'contraband', 'outcome', 'officer_id']
stops12.columns = ['date', 'county', 'gender', 'race',
       'searched', 'contraband', 'outcome', 'officer_id']
stops13.columns = ['date', 'county', 'gender', 'race',
       'searched', 'contraband', 'outcome', 'officer_id']
stops14.columns = ['date', 'county', 'gender', 'race',
       'searched', 'contraband', 'outcome', 'officer_id']
stops15.columns = ['date', 'county', 'gender', 'race',
       'searched', 'contraband', 'outcome', 'officer_id']
# Group by date for total stops per day of year
by_date10 = stops10.groupby("date").count()
by_date10 = by_date10[["county"]]
by_date10.columns = ["stops_2010"]

# Reset index and remove year to leave just month/day
by_date10.reset_index(inplace=True) 
by_date10["date"] = by_date10["date"].map(lambda x: str(x)[5:])

# Group other dfs by date and reset index so they can be joined
by_date11 = stops11.groupby("date").count()
by_date11.reset_index(inplace=True)
by_date11 = by_date11[["county"]]
by_date11.columns = ["stops_2011"]

by_date12 = stops12.groupby("date").count()
by_date12.reset_index(inplace=True)
by_date12 = by_date12[["county"]]
by_date12.columns = ["stops_2012"]

by_date13 = stops13.groupby("date").count()
by_date13.reset_index(inplace=True)
by_date13 = by_date13[["county"]]
by_date13.columns = ["stops_2013"]

by_date14 = stops14.groupby("date").count()
by_date14.reset_index(inplace=True)
by_date14 = by_date14[["county"]]
by_date14.columns = ["stops_2014"]

by_date15 = stops15.groupby("date").count()
by_date15.reset_index(inplace=True)
by_date15 = by_date15[["county"]]
by_date15.columns = ["stops_2015"]
# Add columns together for single dataframe
by_date10["stops_2011"] = by_date11["stops_2011"]
by_date10["stops_2012"] = by_date12["stops_2012"]
by_date10["stops_2013"] = by_date13["stops_2013"]
by_date10["stops_2014"] = by_date14["stops_2014"]
by_date10["stops_2015"] = by_date15["stops_2015"]

# Rename for clarity
total_stops_by_year = by_date10
total_stops_by_year.head()
# Same stuff with grouping by gender
by_gender = stops10.groupby("gender").count()
by_gender = by_gender[["outcome"]]
by_gender.columns = ["stops_2010"]

by_gender11 = stops11.groupby("gender").count()
by_gender11 = by_gender11[["outcome"]]
by_gender11.columns = ["stops_2011"]

by_gender12 = stops12.groupby("gender").count()
by_gender12 = by_gender12[["outcome"]]
by_gender12.columns = ["stops_2012"]

by_gender13 = stops13.groupby("gender").count()
by_gender13 = by_gender13[["outcome"]]
by_gender13.columns = ["stops_2013"]

by_gender14 = stops14.groupby("gender").count()
by_gender14 = by_gender14[["outcome"]]
by_gender14.columns = ["stops_2014"]

by_gender15 = stops15.groupby("gender").count()
by_gender15 = by_gender15[["outcome"]]
by_gender15.columns = ["stops_2015"]
# Joining dataframes together
by_gender["stops_2011"] = by_gender11["stops_2011"]
by_gender["stops_2012"] = by_gender12["stops_2012"]
by_gender["stops_2013"] = by_gender13["stops_2013"]
by_gender["stops_2014"] = by_gender14["stops_2014"]
by_gender["stops_2015"] = by_gender15["stops_2015"]

# Reset index and rename for clarity
by_gender.reset_index(inplace=True)
total_stops_by_gender = by_gender
total_stops_by_gender.head()
# Same stuff with grouping by race
by_race = stops10.groupby("race").count()
by_race = by_race[["outcome"]]
by_race.columns = ["stops_2010"]

by_race11 = stops11.groupby("race").count()
by_race11 = by_race11[["outcome"]]
by_race11.columns = ["stops_2011"]

by_race12 = stops12.groupby("race").count()
by_race12 = by_race12[["outcome"]]
by_race12.columns = ["stops_2012"]

by_race13 = stops13.groupby("race").count()
by_race13 = by_race13[["outcome"]]
by_race13.columns = ["stops_2013"]

by_race14 = stops14.groupby("race").count()
by_race14 = by_race14[["outcome"]]
by_race14.columns = ["stops_2014"]

by_race15 = stops15.groupby("race").count()
by_race15 = by_race15[["outcome"]]
by_race15.columns = ["stops_2015"]
# Joining dataframes together
by_race["stops_2011"] = by_race11["stops_2011"]
by_race["stops_2012"] = by_race12["stops_2012"]
by_race["stops_2013"] = by_race13["stops_2013"]
by_race["stops_2014"] = by_race14["stops_2014"]
by_race["stops_2015"] = by_race15["stops_2015"]

# Reset index and rename for clarity
by_race.reset_index(inplace=True)
total_stops_by_race = by_race
total_stops_by_race.head()
# Same stuff with grouping by county
by_county = stops10.groupby("county").count()
by_county = by_county[["outcome"]]
by_county.columns = ["stops_2010"]

by_county11 = stops11.groupby("county").count()
by_county11 = by_county11[["outcome"]]
by_county11.columns = ["stops_2011"]

by_county12 = stops12.groupby("county").count()
by_county12 = by_county12[["outcome"]]
by_county12.columns = ["stops_2012"]

by_county13 = stops13.groupby("county").count()
by_county13 = by_county13[["outcome"]]
by_county13.columns = ["stops_2013"]

by_county14 = stops14.groupby("county").count()
by_county14 = by_county14[["outcome"]]
by_county14.columns = ["stops_2014"]

by_county15 = stops15.groupby("county").count()
by_county15 = by_county15[["outcome"]]
by_county15.columns = ["stops_2015"]
# Joining dataframes together
by_county["stops_2011"] = by_county11["stops_2011"]
by_county["stops_2012"] = by_county12["stops_2012"]
by_county["stops_2013"] = by_county13["stops_2013"]
by_county["stops_2014"] = by_county14["stops_2014"]
by_county["stops_2015"] = by_county15["stops_2015"]

# Reset index and rename for clarity
by_county.reset_index(inplace=True)
total_stops_by_county = by_county
total_stops_by_county.head()

# Import SQLAlchemy dependencies
import sqlalchemy
from sqlalchemy import create_engine, MetaData, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Numeric, Text, Float, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship
engine = create_engine("sqlite:///speeding.sqlite")
Base = declarative_base()
class Date_stops(Base):
    
    __tablename__ = 'date_stops'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Text)
    stops_2010 = Column(Integer)
    stops_2011 = Column(Integer)
    stops_2012 = Column(Integer)
    stops_2013 = Column(Integer)
    stops_2014 = Column(Integer)
    stops_2015 = Column(Integer)
    
    def __repr__(self):
        return f"{self.date}: 2010 - {self.stops_2010}"
    
class Gender_stops(Base):
    
    __tablename__ = 'gender_stops'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    gender = Column(Text)
    stops_2010 = Column(Integer)
    stops_2011 = Column(Integer)
    stops_2012 = Column(Integer)
    stops_2013 = Column(Integer)
    stops_2014 = Column(Integer)
    stops_2015 = Column(Integer)
    
    def __repr__(self):
        return f"{self.gender}: 2010 - {self.stops_2010}"

class Race_stops(Base):
    
    __tablename__ = 'race_stops'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    race = Column(Text)
    stops_2010 = Column(Integer)
    stops_2011 = Column(Integer)
    stops_2012 = Column(Integer)
    stops_2013 = Column(Integer)
    stops_2014 = Column(Integer)
    stops_2015 = Column(Integer)
    
    def __repr__(self):
        return f"{self.race}: 2010 - {self.stops_2010}"

class County_stops(Base):
    
    __tablename__ = 'county_stops'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    county = Column(Text)
    stops_2010 = Column(Integer)
    stops_2011 = Column(Integer)
    stops_2012 = Column(Integer)
    stops_2013 = Column(Integer)
    stops_2014 = Column(Integer)
    stops_2015 = Column(Integer)
    
    def __repr__(self):
        return f"{self.county}: 2010 - {self.stops_2010}"
Base.metadata.create_all(engine)
engine.table_names()
inspector = inspect(engine)
inspector.get_columns('county_stops')
# connect to the database
conn = engine.connect()

# Orient='records' creates a list of data to write
date_data = total_stops_by_year.to_dict(orient='records')
gender_data = total_stops_by_gender.to_dict(orient='records')
race_data = total_stops_by_race.to_dict(orient='records')
county_data = total_stops_by_county.to_dict(orient='records')

# Insert the dataframe into the database in one bulk insert
conn.execute(Date_stops.__table__.insert(), date_data)
conn.execute(Gender_stops.__table__.insert(), gender_data)
conn.execute(Race_stops.__table__.insert(), race_data)
conn.execute(County_stops.__table__.insert(), county_data)

conn.close()
engine.execute("SELECT * FROM date_stops LIMIT 5").fetchall()
engine.execute("SELECT * FROM gender_stops LIMIT 5").fetchall()
engine.execute("SELECT * FROM race_stops LIMIT 5").fetchall()
engine.execute("SELECT * FROM county_stops LIMIT 5").fetchall()

