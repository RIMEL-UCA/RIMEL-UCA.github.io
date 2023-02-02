# Import initial dependencies
import os
import pandas as pd
import numpy as np
# Create OS paths for each data set
y2010 = os.path.join("Texas Speed Related Incidents Data Set","TX 2010 Speed Related Crashes Data.csv")
y2011 = os.path.join("Texas Speed Related Incidents Data Set","TX 2011 Speed Related Crashes Data.csv")
y2012 = os.path.join("Texas Speed Related Incidents Data Set","TX 2012 Speed Related Crashes Data.csv")
y2013 = os.path.join("Texas Speed Related Incidents Data Set","TX 2013 Speed Related Crashes Data.csv")
y2014 = os.path.join("Texas Speed Related Incidents Data Set","TX 2014 Speed Related Crashes Data.csv")
y2015 = os.path.join("Texas Speed Related Incidents Data Set","TX 2015 Speed Related Crashes Data.csv")
y2016 = os.path.join("Texas Speed Related Incidents Data Set","TX 2016 Speed Related Crashes Data.csv")
y2017 = os.path.join("Texas Speed Related Incidents Data Set","TX 2017 Speed Related Crashes Data.csv")
y2018 = os.path.join("Texas Speed Related Incidents Data Set","TX 2018 Speed Related Crashes Data.csv")

# Create data frames for each data set via read_csv
y2010_df = pd.read_csv(y2010, encoding = "utf-8", low_memory = False)
y2011_df = pd.read_csv(y2011, encoding = "utf-8", low_memory = False)
y2012_df = pd.read_csv(y2012, encoding = "utf-8", low_memory = False)
y2013_df = pd.read_csv(y2013, encoding = "utf-8", low_memory = False)
y2014_df = pd.read_csv(y2014, encoding = "utf-8", low_memory = False)
y2015_df = pd.read_csv(y2015, encoding = "utf-8", low_memory = False)
y2016_df = pd.read_csv(y2016, encoding = "utf-8", low_memory = False)
y2017_df = pd.read_csv(y2017, encoding = "utf-8", low_memory = False)
y2018_df = pd.read_csv(y2018, encoding = "utf-8", low_memory = False)
y2018_df.head()
# Append all the data sets into a single data set
speedlimit_crash_df = y2010_df.append(y2011_df)
speedlimit_crash_df = speedlimit_crash_df.append(y2012_df)
speedlimit_crash_df = speedlimit_crash_df.append(y2013_df)
speedlimit_crash_df = speedlimit_crash_df.append(y2014_df)
speedlimit_crash_df = speedlimit_crash_df.append(y2015_df)
speedlimit_crash_df = speedlimit_crash_df.append(y2016_df)
speedlimit_crash_df = speedlimit_crash_df.append(y2017_df)
speedlimit_crash_df = speedlimit_crash_df.append(y2018_df)
speedlimit_crash_df.head()
# Drop rows from teh data set with duplicate Crash ID's
speedlimit_crash_df = speedlimit_crash_df.drop_duplicates(['Crash ID'], keep='first')
speedlimit_crash_df = speedlimit_crash_df.reset_index(drop=True)
speedlimit_crash_df.head(2)
# Rename column names to be ORM class/table friendly
speedlimit_crash_df = speedlimit_crash_df.rename(columns={'Crash ID': 'crash_id','Crash Death Count': 'crash_death_count', 'Crash Severity': 'crash_severity', 'Crash Time': 'crash_time', 'Crash Total Injury Count': 'crash_total_injury_count', 'Crash Year': 'crash_year', 'Day of Week': 'day_of_week', 'Manner of Collision': 'manner_of_collision', 'Population Group': 'population_group', 'Road Class': 'road_class', 'Speed Limit': 'speed_limit', 'Weather Condition': 'weather_condition', 'Vehicle Color': 'vehicle_color', 'Person Age': 'person_age', 'Person Ethnicity': 'person_ethnicity', 'Person Gender': 'person_gender', 'Person Type': 'person_type'})

# Turn all columns headers to lowercase
speedlimit_crash_df.columns = speedlimit_crash_df.columns.str.lower()

speedlimit_crash_df.head(20)
# Replace 
for column in speedlimit_crash_df.columns[1:]:
    if (column == "crash_death_count") or (column == "crash_time") or (column == "crash_total_injury_count") or (column == "crash_year") or (column == "latitude") or (column == "longitude") or (column == "speed_limit") or (column == "person_age"):
         speedlimit_crash_df[column] = speedlimit_crash_df[column].replace("No Data", 0)
    else:
        speedlimit_crash_df[column] = speedlimit_crash_df[column].replace("No Data", "")

#speedlimit_crash_df.iloc[:,0:11]
speedlimit_crash_df.head()
# Number of rows in the df
len(speedlimit_crash_df.index)
# Create new data frame with crash totals by gender

by_gender = y2010_df[y2010_df["Person Gender"] != "Unknown"]
by_gender = by_gender[by_gender["Person Gender"] != "No Data"]
by_gender = by_gender.groupby("Person Gender").count()
by_gender = by_gender[["Crash ID"]]
by_gender.columns = ["crashes_2010"]

by_gender11 = y2011_df[y2011_df["Person Gender"] != "Unknown"]
by_gender11 = by_gender11[by_gender11["Person Gender"] != "No Data"]
by_gender11 = by_gender11.groupby("Person Gender").count()
by_gender11 = by_gender11[["Crash ID"]]
by_gender11.columns = ["crashes_2011"]

by_gender12 = y2012_df[y2012_df["Person Gender"] != "Unknown"]
by_gender12 = by_gender12[by_gender12["Person Gender"] != "No Data"]
by_gender12 = by_gender12.groupby("Person Gender").count()
by_gender12 = by_gender12[["Crash ID"]]
by_gender12.columns = ["crashes_2012"]

by_gender13 = y2013_df[y2013_df["Person Gender"] != "Unknown"]
by_gender13 = by_gender13[by_gender13["Person Gender"] != "No Data"]
by_gender13 = by_gender13.groupby("Person Gender").count()
by_gender13 = by_gender13[["Crash ID"]]
by_gender13.columns = ["crashes_2013"]

by_gender14 = y2014_df[y2014_df["Person Gender"] != "Unknown"]
by_gender14 = by_gender14[by_gender14["Person Gender"] != "No Data"]
by_gender14 = by_gender14.groupby("Person Gender").count()
by_gender14 = by_gender14[["Crash ID"]]
by_gender14.columns = ["crashes_2014"]

by_gender15 = y2015_df[y2015_df["Person Gender"] != "Unknown"]
by_gender15 = by_gender15[by_gender15["Person Gender"] != "No Data"]
by_gender15 = by_gender15.groupby("Person Gender").count()
by_gender15 = by_gender15[["Crash ID"]]
by_gender15.columns = ["crashes_2015"]

by_gender16 = y2016_df[y2016_df["Person Gender"] != "Unknown"]
by_gender16 = by_gender16[by_gender16["Person Gender"] != "No Data"]
by_gender16 = by_gender16.groupby("Person Gender").count()
by_gender16 = by_gender16[["Crash ID"]]
by_gender16.columns = ["crashes_2016"]

by_gender17 = y2017_df[y2017_df["Person Gender"] != "Unknown"]
by_gender17 = by_gender17[by_gender17["Person Gender"] != "No Data"]
by_gender17 = by_gender17.groupby("Person Gender").count()
by_gender17 = by_gender17[["Crash ID"]]
by_gender17.columns = ["crashes_2017"]

by_gender18 = y2018_df[y2018_df["Person Gender"] != "Unknown"]
by_gender18 = by_gender18[by_gender18["Person Gender"] != "No Data"]
by_gender18 = by_gender18.groupby("Person Gender").count()
by_gender18 = by_gender18[["Crash ID"]]
by_gender18.columns = ["crashes_2018"]

by_gender16
# Joining dataframes together
by_gender["crashes_2011"] = by_gender11["crashes_2011"]
by_gender["crashes_2012"] = by_gender12["crashes_2012"]
by_gender["crashes_2013"] = by_gender13["crashes_2013"]
by_gender["crashes_2014"] = by_gender14["crashes_2014"]
by_gender["crashes_2015"] = by_gender15["crashes_2015"]
by_gender["crashes_2016"] = by_gender16["crashes_2016"]
by_gender["crashes_2017"] = by_gender17["crashes_2017"]
by_gender["crashes_2018"] = by_gender18["crashes_2018"]

# Reset index and rename for clarity
by_gender.reset_index(inplace=True)
by_gender.head()
by_gender = by_gender.transpose()
by_gender.reset_index(inplace=True)
by_gender.columns = ["year", "female", "male"]
by_gender = by_gender.drop(by_gender.index[0])
by_gender["year"] = by_gender["year"].map(lambda x: str(x)[8:])
by_gender
# Rename the df
total_crashes_by_gender = by_gender
# Same df creation process with grouping by race
by_race = y2010_df[y2010_df["Person Ethnicity"] != "Unknown"]
by_race = by_race[by_race["Person Ethnicity"] != "No Data"]
by_race = by_race.groupby("Person Ethnicity").count()
by_race = by_race[["Crash ID"]]
by_race.columns = ["crashes_2010"]

by_race11 = y2011_df[y2011_df["Person Ethnicity"] != "Unknown"]
by_race11 = by_race11[by_race11["Person Ethnicity"] != "No Data"]
by_race11 = by_race11.groupby("Person Ethnicity").count()
by_race11 = by_race11[["Crash ID"]]
by_race11.columns = ["crashes_2011"]

by_race12 = y2012_df[y2012_df["Person Ethnicity"] != "Unknown"]
by_race12 = by_race12[by_race12["Person Ethnicity"] != "No Data"]
by_race12 = by_race12.groupby("Person Ethnicity").count()
by_race12 = by_race12[["Crash ID"]]
by_race12.columns = ["crashes_2012"]

by_race13 = y2013_df[y2013_df["Person Ethnicity"] != "Unknown"]
by_race13 = by_race13[by_race13["Person Ethnicity"] != "No Data"]
by_race13 = by_race13.groupby("Person Ethnicity").count()
by_race13 = by_race13[["Crash ID"]]
by_race13.columns = ["crashes_2013"]

by_race14 = y2014_df[y2014_df["Person Ethnicity"] != "Unknown"]
by_race14 = by_race14[by_race14["Person Ethnicity"] != "No Data"]
by_race14 = by_race14.groupby("Person Ethnicity").count()
by_race14 = by_race14[["Crash ID"]]
by_race14.columns = ["crashes_2014"]

by_race15 = y2015_df[y2015_df["Person Ethnicity"] != "Unknown"]
by_race15 = by_race15[by_race15["Person Ethnicity"] != "No Data"]
by_race15 = by_race15.groupby("Person Ethnicity").count()
by_race15 = by_race15[["Crash ID"]]
by_race15.columns = ["crashes_2015"]

by_race16 = y2016_df[y2016_df["Person Ethnicity"] != "Unknown"]
by_race16 = by_race16[by_race16["Person Ethnicity"] != "No Data"]
by_race16 = by_race16.groupby("Person Ethnicity").count()
by_race16 = by_race16[["Crash ID"]]
by_race16.columns = ["crashes_2016"]

by_race17 = y2017_df[y2017_df["Person Ethnicity"] != "Unknown"]
by_race17 = by_race17[by_race17["Person Ethnicity"] != "No Data"]
by_race17 = by_race17.groupby("Person Ethnicity").count()
by_race17 = by_race17[["Crash ID"]]
by_race17.columns = ["crashes_2017"]

by_race18 = y2018_df[y2018_df["Person Ethnicity"] != "Unknown"]
by_race18 = by_race18[by_race18["Person Ethnicity"] != "No Data"]
by_race18 = by_race18.groupby("Person Ethnicity").count()
by_race18 = by_race18[["Crash ID"]]
by_race18.columns = ["crashes_2018"]


by_race16
# Joining dataframes together
by_race["crashes_2011"] = by_race11["crashes_2011"]
by_race["crashes_2012"] = by_race12["crashes_2012"]
by_race["crashes_2013"] = by_race13["crashes_2013"]
by_race["crashes_2014"] = by_race14["crashes_2014"]
by_race["crashes_2015"] = by_race15["crashes_2015"]
by_race["crashes_2016"] = by_race16["crashes_2016"]
by_race["crashes_2017"] = by_race17["crashes_2017"]
by_race["crashes_2018"] = by_race18["crashes_2018"]

# Reset index and rename for clarity
by_race.reset_index(inplace=True)
by_race
by_race = by_race.transpose()
by_race.reset_index(inplace=True)
by_race.columns = ["year", "native_american", "asian", "black", "hispanic", "other", "white"]
by_race = by_race.drop(by_race.index[0])
by_race["year"] = by_race["year"].map(lambda x: str(x)[8:])
by_race
# Rename df
total_crashes_by_race = by_race
# Same df creation process with grouping by race
by_county = y2010_df[y2010_df["County"] != "Unknown"]
by_county = by_county[by_county["County"] != "No Data"]
by_county = by_county.groupby("County").count()
by_county = by_county[["Crash ID"]]
by_county.columns = ["crashes_2010"]

by_county11 = y2011_df[y2011_df["County"] != "Unknown"]
by_county11 = by_county11[by_county11["County"] != "No Data"]
by_county11 = by_county11.groupby("County").count()
by_county11 = by_county11[["Crash ID"]]
by_county11.columns = ["crashes_2011"]

by_county12 = y2012_df[y2012_df["County"] != "Unknown"]
by_county12 = by_county12[by_county12["County"] != "No Data"]
by_county12 = by_county12.groupby("County").count()
by_county12 = by_county12[["Crash ID"]]
by_county12.columns = ["crashes_2012"]

by_county13 = y2013_df[y2013_df["County"] != "Unknown"]
by_county13 = by_county13[by_county13["County"] != "No Data"]
by_county13 = by_county13.groupby("County").count()
by_county13 = by_county13[["Crash ID"]]
by_county13.columns = ["crashes_2013"]

by_county14 = y2014_df[y2014_df["County"] != "Unknown"]
by_county14 = by_county14[by_county14["County"] != "No Data"]
by_county14 = by_county14.groupby("County").count()
by_county14 = by_county14[["Crash ID"]]
by_county14.columns = ["crashes_2014"]

by_county15 = y2015_df[y2015_df["County"] != "Unknown"]
by_county15 = by_county15[by_county15["County"] != "No Data"]
by_county15 = by_county15.groupby("County").count()
by_county15 = by_county15[["Crash ID"]]
by_county15.columns = ["crashes_2015"]

by_county16 = y2016_df[y2016_df["County"] != "Unknown"]
by_county16 = by_county16[by_county16["County"] != "No Data"]
by_county16 = by_county16.groupby("County").count()
by_county16 = by_county16[["Crash ID"]]
by_county16.columns = ["crashes_2016"]

by_county17 = y2017_df[y2017_df["County"] != "Unknown"]
by_county17 = by_county17[by_county17["County"] != "No Data"]
by_county17 = by_county17.groupby("County").count()
by_county17 = by_county17[["Crash ID"]]
by_county17.columns = ["crashes_2017"]

by_county18 = y2018_df[y2018_df["County"] != "Unknown"]
by_county18 = by_county18[by_county18["County"] != "No Data"]
by_county18 = by_county18.groupby("County").count()
by_county18 = by_county18[["Crash ID"]]
by_county18.columns = ["crashes_2018"]

by_county17.head()
# Joining dataframes together
by_county["crashes_2011"] = by_county11["crashes_2011"]
by_county["crashes_2012"] = by_county12["crashes_2012"]
by_county["crashes_2013"] = by_county13["crashes_2013"]
by_county["crashes_2014"] = by_county14["crashes_2014"]
by_county["crashes_2015"] = by_county15["crashes_2015"]
by_county["crashes_2016"] = by_county16["crashes_2016"]
by_county["crashes_2017"] = by_county17["crashes_2017"]
by_county["crashes_2018"] = by_county18["crashes_2018"]

# Reset index and rename for clarity
by_county.reset_index(inplace=True)
by_county.columns = by_county.columns.str.lower()
by_county.head()
# Rename df
total_crashes_by_county = by_county
# Import SQLAlchemy dependencies
import sqlalchemy
from sqlalchemy import create_engine, MetaData, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Numeric, Text, Float, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship
# Create Engine
engine = create_engine("sqlite:///speeding.sqlite")
# Use `declarative_base` from SQLAlchemy to model the 'crashes' table as an ORM class
# Declare a Base object here
Base = declarative_base()
# Define the ORM class for `Crashes`
class Crashes(Base):
    
    __tablename__ = 'crashes'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    crash_id = Column(Integer, unique=True)
    agency = Column(Text)
    city = Column(Text)
    county = Column(Text)
    crash_death_count = Column(Integer)
    crash_severity = Column(Text)
    crash_time = Column(Integer)
    crash_total_injury_count = Column(Integer)
    crash_year = Column(Integer)
    day_of_week = Column(Text)
    latitude = Column(Float)
    longitude = Column(Float)
    manner_of_collision = Column(Text)
    population_group = Column(Text)
    road_class = Column(Text)
    speed_limit = Column(Integer)
    weather_condition = Column(Text)
    vehicle_color = Column(Text)
    person_age = Column(Integer)
    person_ethnicity = Column(Text)
    person_gender = Column(Text)
    person_type = Column(Text)
    
    
    def __repr__(self):
        return f"id={self.crash_id}, name={self.agency}"
class Gender_crashes(Base):
    
    __tablename__ = 'gender_crashes'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    year = Column(Text)
    female = Column(Integer)
    male = Column(Integer)
    
    def __repr__(self):
        return f"{self.gender}: 2010 - {self.stops_2010}"

class Race_crashes(Base):
    
    __tablename__ = 'race_crashes'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    year = Column(Text)
    native_american = Column(Integer)
    asian = Column(Integer)
    black = Column(Integer)
    hispanic = Column(Integer)
    other = Column(Integer)
    white = Column(Integer)
    
    def __repr__(self):
        return f"{self.race}: 2010 - {self.stops_2010}"

class County_crashes(Base):
    
    __tablename__ = 'county_crashes'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    county = Column(Text)
    crashes_2010 = Column(Integer)
    crashes_2011 = Column(Integer)
    crashes_2012 = Column(Integer)
    crashes_2013 = Column(Integer)
    crashes_2014 = Column(Integer)
    crashes_2015 = Column(Integer)
    crashes_2016 = Column(Integer)
    crashes_2017 = Column(Integer)
    crashes_2018 = Column(Integer)
    
    def __repr__(self):
        return f"{self.county}: 2010 - {self.stops_2010}"
# Use `create_all` to create the tables
Base.metadata.create_all(engine)

# Verify that the table names exist in the database
engine.table_names()
inspector = inspect(engine)
inspector.get_columns('race_crashes')
# connect to the database
conn = engine.connect()

# Orient ='records' creates a list of data to write
gender_data = total_crashes_by_gender.to_dict(orient='records')
race_data = total_crashes_by_race.to_dict(orient='records')
county_data = total_crashes_by_county.to_dict(orient='records')

# Insert the dataframe into the database in one bulk insert
conn.execute(Gender_crashes.__table__.delete())
conn.execute(Gender_crashes.__table__.insert(), gender_data)
conn.execute(Race_crashes.__table__.delete())
conn.execute(Race_crashes.__table__.insert(), race_data)
conn.execute(County_crashes.__table__.delete())
conn.execute(County_crashes.__table__.insert(), county_data)

conn.close()
# Testing the county_crashes table via engine 
engine.execute("SELECT * FROM county_crashes LIMIT 10").fetchall()
# Use Pandas to Bulk insert each data frame into their appropriate table
def populate_table(engine, table):
    """Populates a table from a Pandas DataFrame."""
    
    # connect to the database
    conn = engine.connect()
    
    # Orient='records' creates a list of data to write
    data = speedlimit_crash_df.to_dict(orient='records')

    # Optional: Delete all rows in the table 
    conn.execute(table.delete())

    # Insert the dataframe into the database in one bulk insert
    conn.execute(table.insert(), data)
    conn.close()
    
# Call the function to insert the data for each table
populate_table(engine, Crashes.__table__)
# Use a basic query to validate that the data was inserted correctly for table `crashes`
engine.execute("SELECT * FROM crashes LIMIT 5").fetchall()
# Count the number of records in the 'crashes' table
engine.execute("SELECT COUNT(*) FROM crashes").fetchall()
