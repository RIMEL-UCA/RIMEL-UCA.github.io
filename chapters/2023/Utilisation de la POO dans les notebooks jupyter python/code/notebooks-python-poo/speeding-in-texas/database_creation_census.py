# Import initial dependencies
import os
import pandas as pd
import numpy as np
# Create OS paths for each data set
cen2010 = os.path.join("Texas Census Demographic Data Sets","ACS_10_5YR_DP05_with_ann.csv")
cen2011 = os.path.join("Texas Census Demographic Data Sets","ACS_11_5YR_DP05_with_ann.csv")
cen2012 = os.path.join("Texas Census Demographic Data Sets","ACS_12_5YR_DP05_with_ann.csv")
cen2013 = os.path.join("Texas Census Demographic Data Sets","ACS_13_5YR_DP05_with_ann.csv")
cen2014 = os.path.join("Texas Census Demographic Data Sets","ACS_14_5YR_DP05_with_ann.csv")
cen2015 = os.path.join("Texas Census Demographic Data Sets","ACS_15_5YR_DP05_with_ann.csv")
cen2016 = os.path.join("Texas Census Demographic Data Sets","ACS_16_5YR_DP05_with_ann.csv")

# Create data frames for each data set via read_csv
cen2010_df = pd.read_csv(cen2010, encoding = "utf-8")
cen2011_df = pd.read_csv(cen2011, encoding = "utf-8")
cen2012_df = pd.read_csv(cen2012, encoding = "utf-8")
cen2013_df = pd.read_csv(cen2013, encoding = "utf-8")
cen2014_df = pd.read_csv(cen2014, encoding = "utf-8")
cen2015_df = pd.read_csv(cen2015, encoding = "utf-8")
cen2016_df = pd.read_csv(cen2016, encoding = "utf-8")


cen2014_df
# Drop columns with 'Margins of Error' and 'Percent', since df's are only supposed to contain estimates
# Rename columns to be ORM table friendly and simplification

# df_list = [cen2010_df, cen2011_df, cen2012_df, cen2013_df, cen2014_df, cen2015_df, cen2016_df]

# for df in df_list:
#     df = df.drop(df.filter(like='Margin').columns, 1)
#     df = df.drop(df.filter(like='Percent').columns, 1)
#     df = df.drop(df.filter(like='Id').columns, 1)

# cen2010_df

cen2010_df = cen2010_df.drop(cen2010_df.filter(like='Margin').columns, 1)
cen2010_df = cen2010_df.drop(cen2010_df.filter(like='Percent').columns, 1)
cen2010_df = cen2010_df.drop(cen2010_df.filter(like='Id').columns, 1)

cen2010_df = cen2010_df.rename(columns = {cen2010_df.columns[0]: 'year', cen2010_df.columns[1]: 'total_population',
                                         cen2010_df.columns[2]: 'male', cen2010_df.columns[3]: 'female',
                                         cen2010_df.columns[4]: 'age_under_5', cen2010_df.columns[5]: 'age_5_to_9',
                                         cen2010_df.columns[6]: 'age_10_to_14', cen2010_df.columns[7]: 'age_15_to_19',
                                         cen2010_df.columns[8]: 'age_20_to_24', cen2010_df.columns[9]: 'age_25_to_34',
                                         cen2010_df.columns[10]: 'age_35_to_44', cen2010_df.columns[11]: 'age_45_to_54',
                                         cen2010_df.columns[12]: 'age_55_to_60', cen2010_df.columns[13]: 'age_60_to_64',
                                         cen2010_df.columns[14]: 'age_65_to_74', cen2010_df.columns[15]: 'age_75_to_84',
                                         cen2010_df.columns[16]: 'age_85_and_over', cen2010_df.columns[17]: 'white',
                                         cen2010_df.columns[18]: 'black', cen2010_df.columns[19]: 'native_american',
                                          cen2010_df.columns[20]: 'asian', cen2010_df.columns[21]: 'pacific_islander',
                                          cen2010_df.columns[22]: 'other_race', cen2010_df.columns[23]: 'hispanic'})

# Change the 'geography' column to year column. Assign the year as the row value
cen2010_df["year"] = 2010

cen2011_df = cen2011_df.drop(cen2011_df.filter(like='Margin').columns, 1)
cen2011_df = cen2011_df.drop(cen2011_df.filter(like='Percent').columns, 1)
cen2011_df = cen2011_df.drop(cen2011_df.filter(like='Id').columns, 1)

cen2011_df = cen2011_df.rename(columns = {cen2011_df.columns[0]: 'year', cen2011_df.columns[1]: 'total_population',
                                         cen2011_df.columns[2]: 'male', cen2011_df.columns[3]: 'female',
                                         cen2011_df.columns[4]: 'age_under_5', cen2011_df.columns[5]: 'age_5_to_9',
                                         cen2011_df.columns[6]: 'age_10_to_14', cen2011_df.columns[7]: 'age_15_to_19',
                                         cen2011_df.columns[8]: 'age_20_to_24', cen2011_df.columns[9]: 'age_25_to_34',
                                         cen2011_df.columns[10]: 'age_35_to_44', cen2011_df.columns[11]: 'age_45_to_54',
                                         cen2011_df.columns[12]: 'age_55_to_60', cen2011_df.columns[13]: 'age_60_to_64',
                                         cen2011_df.columns[14]: 'age_65_to_74', cen2011_df.columns[15]: 'age_75_to_84',
                                         cen2011_df.columns[16]: 'age_85_and_over', cen2011_df.columns[17]: 'white',
                                         cen2011_df.columns[18]: 'black', cen2011_df.columns[19]: 'native_american',
                                          cen2011_df.columns[20]: 'asian', cen2011_df.columns[21]: 'pacific_islander',
                                          cen2011_df.columns[22]: 'other_race', cen2011_df.columns[23]: 'hispanic'})

# Change the 'geography' column to year column. Assign the year as the row value
cen2011_df["year"] = 2011

cen2012_df = cen2012_df.drop(cen2012_df.filter(like='Margin').columns, 1)
cen2012_df = cen2012_df.drop(cen2012_df.filter(like='Percent').columns, 1)
cen2012_df = cen2012_df.drop(cen2012_df.filter(like='Id').columns, 1)

cen2012_df = cen2012_df.rename(columns = {cen2012_df.columns[0]: 'year', cen2012_df.columns[1]: 'total_population',
                                         cen2012_df.columns[2]: 'male', cen2012_df.columns[3]: 'female',
                                         cen2012_df.columns[4]: 'age_under_5', cen2012_df.columns[5]: 'age_5_to_9',
                                         cen2012_df.columns[6]: 'age_10_to_14', cen2012_df.columns[7]: 'age_15_to_19',
                                         cen2012_df.columns[8]: 'age_20_to_24', cen2012_df.columns[9]: 'age_25_to_34',
                                         cen2012_df.columns[10]: 'age_35_to_44', cen2012_df.columns[11]: 'age_45_to_54',
                                         cen2012_df.columns[12]: 'age_55_to_60', cen2012_df.columns[13]: 'age_60_to_64',
                                         cen2012_df.columns[14]: 'age_65_to_74', cen2012_df.columns[15]: 'age_75_to_84',
                                         cen2012_df.columns[16]: 'age_85_and_over', cen2012_df.columns[17]: 'white',
                                         cen2012_df.columns[18]: 'black', cen2012_df.columns[19]: 'native_american',
                                          cen2012_df.columns[20]: 'asian', cen2012_df.columns[21]: 'pacific_islander',
                                          cen2012_df.columns[22]: 'other_race', cen2012_df.columns[23]: 'hispanic'})

cen2012_df["year"] = 2012

cen2013_df = cen2013_df.drop(cen2013_df.filter(like='Margin').columns, 1)
cen2013_df = cen2013_df.drop(cen2013_df.filter(like='Percent').columns, 1)
cen2013_df = cen2013_df.drop(cen2013_df.filter(like='Id').columns, 1)

cen2013_df = cen2013_df.rename(columns = {cen2013_df.columns[0]: 'year', cen2013_df.columns[1]: 'total_population',
                                         cen2013_df.columns[2]: 'male', cen2013_df.columns[3]: 'female',
                                         cen2013_df.columns[4]: 'age_under_5', cen2013_df.columns[5]: 'age_5_to_9',
                                         cen2013_df.columns[6]: 'age_10_to_14', cen2013_df.columns[7]: 'age_15_to_19',
                                         cen2013_df.columns[8]: 'age_20_to_24', cen2013_df.columns[9]: 'age_25_to_34',
                                         cen2013_df.columns[10]: 'age_35_to_44', cen2013_df.columns[11]: 'age_45_to_54',
                                         cen2013_df.columns[12]: 'age_55_to_60', cen2013_df.columns[13]: 'age_60_to_64',
                                         cen2013_df.columns[14]: 'age_65_to_74', cen2013_df.columns[15]: 'age_75_to_84',
                                         cen2013_df.columns[16]: 'age_85_and_over', cen2013_df.columns[17]: 'white',
                                         cen2013_df.columns[18]: 'black', cen2013_df.columns[19]: 'native_american',
                                          cen2013_df.columns[20]: 'asian', cen2013_df.columns[21]: 'pacific_islander',
                                          cen2013_df.columns[22]: 'other_race', cen2013_df.columns[23]: 'hispanic'})

cen2013_df["year"] = 2013

cen2014_df = cen2014_df.drop(cen2014_df.filter(like='Margin').columns, 1)
cen2014_df = cen2014_df.drop(cen2014_df.filter(like='Percent').columns, 1)
cen2014_df = cen2014_df.drop(cen2014_df.filter(like='Id').columns, 1)

cen2014_df = cen2014_df.rename(columns = {cen2014_df.columns[0]: 'year', cen2014_df.columns[1]: 'total_population',
                                         cen2014_df.columns[2]: 'male', cen2014_df.columns[3]: 'female',
                                         cen2014_df.columns[4]: 'age_under_5', cen2014_df.columns[5]: 'age_5_to_9',
                                         cen2014_df.columns[6]: 'age_10_to_14', cen2014_df.columns[7]: 'age_15_to_19',
                                         cen2014_df.columns[8]: 'age_20_to_24', cen2014_df.columns[9]: 'age_25_to_34',
                                         cen2014_df.columns[10]: 'age_35_to_44', cen2014_df.columns[11]: 'age_45_to_54',
                                         cen2014_df.columns[12]: 'age_55_to_60', cen2014_df.columns[13]: 'age_60_to_64',
                                         cen2014_df.columns[14]: 'age_65_to_74', cen2014_df.columns[15]: 'age_75_to_84',
                                         cen2014_df.columns[16]: 'age_85_and_over', cen2014_df.columns[17]: 'white',
                                         cen2014_df.columns[18]: 'black', cen2014_df.columns[19]: 'native_american',
                                          cen2014_df.columns[20]: 'asian', cen2014_df.columns[21]: 'pacific_islander',
                                          cen2014_df.columns[22]: 'other_race', cen2014_df.columns[23]: 'hispanic'})

cen2014_df["year"] = 2014

cen2015_df = cen2015_df.drop(cen2015_df.filter(like='Margin').columns, 1)
cen2015_df = cen2015_df.drop(cen2015_df.filter(like='Percent').columns, 1)
cen2015_df = cen2015_df.drop(cen2015_df.filter(like='Id').columns, 1)

cen2015_df = cen2015_df.rename(columns = {cen2015_df.columns[0]: 'year', cen2015_df.columns[1]: 'total_population',
                                         cen2015_df.columns[2]: 'male', cen2015_df.columns[3]: 'female',
                                         cen2015_df.columns[4]: 'age_under_5', cen2015_df.columns[5]: 'age_5_to_9',
                                         cen2015_df.columns[6]: 'age_10_to_14', cen2015_df.columns[7]: 'age_15_to_19',
                                         cen2015_df.columns[8]: 'age_20_to_24', cen2015_df.columns[9]: 'age_25_to_34',
                                         cen2015_df.columns[10]: 'age_35_to_44', cen2015_df.columns[11]: 'age_45_to_54',
                                         cen2015_df.columns[12]: 'age_55_to_60', cen2015_df.columns[13]: 'age_60_to_64',
                                         cen2015_df.columns[14]: 'age_65_to_74', cen2015_df.columns[15]: 'age_75_to_84',
                                         cen2015_df.columns[16]: 'age_85_and_over', cen2015_df.columns[17]: 'white',
                                         cen2015_df.columns[18]: 'black', cen2015_df.columns[19]: 'native_american',
                                          cen2015_df.columns[20]: 'asian', cen2015_df.columns[21]: 'pacific_islander',
                                          cen2015_df.columns[22]: 'other_race', cen2015_df.columns[23]: 'hispanic'})

cen2015_df["year"] = 2015

cen2016_df = cen2016_df.drop(cen2016_df.filter(like='Margin').columns, 1)
cen2016_df = cen2016_df.drop(cen2016_df.filter(like='Percent').columns, 1)
cen2016_df = cen2016_df.drop(cen2016_df.filter(like='Id').columns, 1)

cen2016_df = cen2016_df.rename(columns = {cen2016_df.columns[0]: 'year', cen2016_df.columns[1]: 'total_population',
                                         cen2016_df.columns[2]: 'male', cen2016_df.columns[3]: 'female',
                                         cen2016_df.columns[4]: 'age_under_5', cen2016_df.columns[5]: 'age_5_to_9',
                                         cen2016_df.columns[6]: 'age_10_to_14', cen2016_df.columns[7]: 'age_15_to_19',
                                         cen2016_df.columns[8]: 'age_20_to_24', cen2016_df.columns[9]: 'age_25_to_34',
                                         cen2016_df.columns[10]: 'age_35_to_44', cen2016_df.columns[11]: 'age_45_to_54',
                                         cen2016_df.columns[12]: 'age_55_to_60', cen2016_df.columns[13]: 'age_60_to_64',
                                         cen2016_df.columns[14]: 'age_65_to_74', cen2016_df.columns[15]: 'age_75_to_84',
                                         cen2016_df.columns[16]: 'age_85_and_over', cen2016_df.columns[17]: 'white',
                                         cen2016_df.columns[18]: 'black', cen2016_df.columns[19]: 'native_american',
                                          cen2016_df.columns[20]: 'asian', cen2016_df.columns[21]: 'pacific_islander',
                                          cen2016_df.columns[22]: 'other_race', cen2016_df.columns[23]: 'hispanic'})

cen2016_df["year"] = 2016

cen2016_df
# Append all the data sets into a single data set
census_data_df = cen2010_df.append(cen2011_df)
census_data_df = census_data_df.append(cen2012_df)
census_data_df = census_data_df.append(cen2013_df)
census_data_df = census_data_df.append(cen2014_df)
census_data_df = census_data_df.append(cen2015_df)
census_data_df = census_data_df.append(cen2016_df)

census_data_df.head(10)
# Reset the index for the census df
census_data_df = census_data_df.reset_index(drop=True)
census_data_df
## Database engineering

# Import SQLAlchemy dependencies
import sqlalchemy
from sqlalchemy import create_engine, MetaData, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Numeric, Text, Float, ForeignKey, BigInteger
from sqlalchemy.orm import sessionmaker, relationship
# Create Engine
engine = create_engine("sqlite:///speeding.sqlite")
# Use `declarative_base` from SQLAlchemy to model the 'crashes' table as an ORM class
# Declare a Base object here
Base = declarative_base()
# Define the ORM class for `Demographics`
class Demographics(Base):
    
    __tablename__ = 'demographics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    year = Column(Integer)
    total_population = Column(Integer)
    male = Column(Integer)
    female = Column(Integer)
    age_under_5 = Column(Integer)
    age_5_to_9 = Column(Integer)
    age_10_to_14 = Column(Integer)                                
    age_15_to_19 = Column(Integer)
    age_20_to_24 = Column(Integer)
    age_25_to_34 = Column(Integer)
    age_35_to_44 = Column(Integer)
    age_45_to_54 = Column(Integer)
    age_55_to_60 = Column(Integer)
    age_60_to_64 = Column(Integer)
    age_65_to_74 = Column(Integer)
    age_75_to_84 = Column(Integer)
    age_85_and_over = Column(Integer)
    white = Column(Integer)
    black = Column(Integer)
    native_american = Column(Integer)
    asian = Column(Integer)
    pacific_islander = Column(Integer)
    other_race = Column(Integer)
    hispanic = Column(Integer)
            
        
    def __repr__(self):
        return f"id={self.crash_id}, name={self.year}"
# Use `create_all` to create the tables
Base.metadata.create_all(engine)
# Verify that the table names exist in the database
engine.table_names()
inspector = inspect(engine)
inspector.get_columns('demographics')
census_data_df.astype(np.int8).dtypes
pandas_records = census_data_df.to_dict(orient='records')
len(pandas_records)
fixed_records = []

for item in pandas_records:
    fixed_records.append({ key: int(value) for key, value in item.items() })
# Use Pandas to Bulk insert each data frame into their appropriate table
def populate_table(engine, table):
    """Populates a table from a Pandas DataFrame."""
    
    # connect to the database
    conn = engine.connect()
    
    # Orient='records' creates a list of data to write
    data = census_data_df.to_dict(orient='records')

    # Optional: Delete all rows in the table 
    conn.execute(table.delete())

    # Insert the dataframe into the database in one bulk insert
    conn.execute(table.insert(), fixed_records)
    conn.close()
    
# Call the function to insert the data for each table
populate_table(engine, Demographics.__table__)
# Use a basic query to validate that the data was inserted correctly for table `demographics`
engine.execute("SELECT * FROM demographics LIMIT 7").fetchall()


