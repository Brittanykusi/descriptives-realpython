############### IMPORT PACKAGES ######################
import researchpy as rp # Researchpy produces Pandas DataFrames that contains relevant statistical testing information that is commonly required for academic research.
import pandas as pd
######################################################

############### IMPORT DATASET #######################
df = pd.read_csv('https://raw.githubusercontent.com/Brittanykusi/HHA-507-2022/main/descriptive/example1/data/data.csv')
df
######################################################


rp.codebook(df) # Prints out descriptive information for a Pandas Series or DataFrame object.
df.columns # list column names

# rp.summary_cont() - Returns a nice data table as a Pandas DataFrame that includes
# the variable name, total number of non-missing observations, 
# standard deviation, standard error, and the 95% confidence 
# interval.
rp.summary_cont(df[['Age', 'HR', 'sBP']])

# Returns a data table as a Pandas DataFrame that includes the 
# counts and percentages of each category. 
rp.summary_cat(df[['Group', 'Smoke']])

# provides value counts of the column name provided
df['Group'].value_counts()

df['Smoke'].value_counts()