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
y2010_df = pd.read_csv(y2010, encoding = "utf-8")
y2011_df = pd.read_csv(y2011, encoding = "utf-8")
y2012_df = pd.read_csv(y2012, encoding = "utf-8")
y2013_df = pd.read_csv(y2013, encoding = "utf-8")
y2014_df = pd.read_csv(y2014, encoding = "utf-8")
y2015_df = pd.read_csv(y2015, encoding = "utf-8")
y2016_df = pd.read_csv(y2016, encoding = "utf-8")
y2017_df = pd.read_csv(y2017, encoding = "utf-8")
y2018_df = pd.read_csv(y2018, encoding = "utf-8")
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
# speedlimit_crash_df = speedlimit_crash_df.set_index("crash_id")
# speedlimit_crash_df.head()
# Number of rows in the df
len(speedlimit_crash_df.index)
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
# Use `create_all` to create the tables
Base.metadata.create_all(engine)
# speedlimit_crash_df.set_index("crash_id")\
#                     .to_sql(name="crashes", con=engine, if_exists="append")
# Verify that the table names exist in the database
engine.table_names()
inspector = inspect(engine)
inspector.get_columns('crashes')

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

