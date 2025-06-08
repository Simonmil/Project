import os;  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin'] # Found under the additional variable (feature) information

# Explanation of parameters for read_csv:
# na_values are the marker of unavailable values, sep are the separator (delimiter) of the values (here space), skipinitialspace skips the space after delimiter (Here there are more than one space between the values)
# comment gives what the symbol for comments are (ex. # in pyhon) making pandas ignore everything after the symbol (Here car name is ignored)
raw_dataset = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True) 

dataset = raw_dataset.copy()    # make a copy to retain the original data
#print(dataset.tail())          # Print the last 5 datapoints. We can also check that the column names are placed correctly

#print(dataset.isna().sum())     # prints which columns that contains na-values (? in this dataset), and how many are na in each column
dataset = dataset.dropna()      # This removes the rows where there are na-values. 


# NOTE: Origin is categorical where a number is related to a country. We will make them. The values here will therefore we one-hot encoded.

dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'}) # Map the values 1,2,3 to origin locations, so all values are switched to locations.
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='',prefix_sep='') # This changes the one Origin column to the 3 locations and adds true or false values  
#print(dataset.tail())


# Split the data into training and testing sets

train_dataset = dataset.sample(frac=0.8, random_state=0)    # sample 80 % of the data randomly into a dataset we can use for training. random_state is the seed for the random number generator
test_dataset = dataset.drop(train_dataset.index)            # create a dataset for testing by dropping the datapoints present in the training set. 

# Inspect the data

#sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
#plt.show() # Required to see plots in Windows

print(train_dataset.describe().transpose())