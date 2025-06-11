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
"""
print(train_dataset.describe().transpose())
sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
plt.show() # Required to see plots in Windows
"""
# Splitting the label (target value) from the features
# Make copies of the datasets that will contain the features
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG') # The pop function extracts and removes the values under MPG from the feature-datasets and puts them in a new variable
test_labels = test_features.pop('MPG')

# Normalize the features. Good practice to do this as it makes the training more stable.
#print(train_dataset.describe().transpose()[['mean','std']]) # Easy way to see how different the means and standard deviations are.

# Create the layer
normalizer = tf.keras.layers.Normalization(axis=-1)

normalizer.adapt(np.array(train_features))  # Fit to the data
#print(normalizer.mean.numpy())              # Calculate the mean and variance of the data and store them in the layer
"""
# Testing the normalization
first = np.array(train_features[:1]).astype(np.float32)
with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())
"""

# Starting with some Linear regression using one and several variables.
# Single-variable linear regression
horsepower = np.array(train_features['Horsepower'])
#test_horsepower = test_features['Horsepower']
#print(train_features['Horsepower'].shape)

normalizer_horsepower = tf.keras.layers.Normalization(input_shape=[1,],axis=None) # Make a new normalizer for the horsepower feature
normalizer_horsepower.adapt(horsepower)

# Make the model. units are the number of nodes in a layer. Here its 1, meaning there is only one weight to optimize representing the m in y=mx+b

horsepower_model = tf.keras.Sequential([
    normalizer_horsepower,
    tf.keras.layers.Dense(units=1)
])

horsepower_model.summary() # Display the model contents
#print(horsepower_model.predict(horsepower[:10]))

# Configuring the training procedure. loss is the function to optimize, Adam is a stochastic gradient descent, and the learning rate is the size of the steps towards minimizing the loss function
horsepower_model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
  loss='mean_absolute_error'
)

# epochs are the number of times the fit procedure will run over the data. If the batch size is different than the length of the data, the model will be updated multiple times during one epoch.
# validation split is the fraction of the training dataset that will be used to validate the data (this is not the same as providing test values)
history = horsepower_model.fit(
  train_features['Horsepower'],
  train_labels,
  epochs=100,
  # suppress logging.
  verbose=0,
  # Calculate validate results on 20% of the training data
  validation_split=0.2
)


# Put the statistics of the fit into a Pandas DataFrame. This is based on the fit to training and validation data.
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
#print(hist.tail())

def plot_loss(history, modelname=''):
  # This function plots the loss of the fit (the validation)
  plt.plot(history.history['loss'], label='loss ' + modelname)
  plt.plot(history.history['val_loss'], label='val_loss ' + modelname)
  plt.ylim([0,10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)
  #plt.draw()

# Plot the loss
plt.figure(1)
plot_loss(history, 'horsepower')


# Run evaluation for the fit by providing the test data
test_results = {}      # Collect results for later
test_results['horsepower_model'] = horsepower_model.evaluate(
  test_features['Horsepower'],
  test_labels
)

x = tf.linspace(0.0,250,251)
y = horsepower_model.predict(x)

def plot_horsepower(x,y,modelname='',color='k'):
  plt.plot(x,y,color=color,label='Predictions ' + modelname)
  plt.xlabel('Horsepower')
  plt.ylabel('MPG')
  plt.legend()
  plt.draw()


plt.figure(2)
plt.scatter(train_features['Horsepower'],train_labels,label='Data')
plot_horsepower(x,y)


# Multivariate linear regression

linear_model = tf.keras.Sequential([
  normalizer, # Using the normalizer defined earlier
  tf.keras.layers.Dense(units=1)
])


#print(linear_model.predict(train_features[:10]))
#print(linear_model.layers[1].kernel)            # Here we check the kernel weights are of the right shape (9,1). This is the m  in y = mx + b


linear_model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
  loss='mean_absolute_error'
)


history = linear_model.fit(
  train_features,
  train_labels,
  epochs=100,
  # suppress logging.
  verbose=0,
  # Calculate validate results on 20% of the training data
  validation_split=0.2
)
plt.figure(1)
plot_loss(history, 'linear_model')


test_results['linear_model'] = linear_model.evaluate(
  test_features,
  test_labels
)

####----------------------####
# Regression with Deep Neural Network
# Many of the previous stuff will be reused 


def build_and_compile_model(norm):
  model = tf.keras.Sequential([
    norm,
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1) # One output layer as we only want 1 output
  ])

  model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
  return model



# Single variable

dnn_horsepower_model = build_and_compile_model(normalizer_horsepower)
dnn_horsepower_model.summary()

history = dnn_horsepower_model.fit(
  train_features['Horsepower'],
  train_labels,
  validation_split=0.2,
  verbose=0,
  epochs=100
)

plt.figure(1)
plot_loss(history, "horsepower_dnn")


x = tf.linspace(0.0,250,251)
y = dnn_horsepower_model.predict(x)

plt.figure(2)
plot_horsepower(x,y,'dnn','b')

test_results['dnn_horsepower_model'] = dnn_horsepower_model.evaluate(test_features['Horsepower'],test_labels, verbose=0)


# DNN with multiple features

dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

history = dnn_model.fit(
  train_features,
  train_labels,
  validation_split=0.2,
  verbose=0,
  epochs=100
)

plt.figure(1)
plot_loss(history,'dnn_model')

test_results['dnn_model'] = dnn_model.evaluate(test_features,test_labels,verbose=0)

print(pd.DataFrame(test_results, index=['Mean absolute Error [MPG]']).T)



# Make predictons

test_predictions = dnn_model.predict(test_features).flatten() # Flatten to make the result one-dim

plt.figure(3)
a = plt.axes(aspect='equal')
plt.scatter(test_labels,test_predictions)
plt.xlabel('True values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0,50]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims,lims)

plt.figure(4)
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
plt.ylabel('Count')
plt.show()

