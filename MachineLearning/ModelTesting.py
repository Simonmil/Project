import os;  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib as plt

"""
TODO:
    - REMEMBER TO FOCUS ON ONE THING AT A TIME!!!!!! Datastuff THEN model-making ETC!
    - Choose a dataset, either Regress or Classification
    - Manipulate the data, take a look at the data by printing, plotting etc. How should I prep the data: batch, numpy, other?
    - Make an appropriate model. Look at the documentation for the different options. Could start with Sequential...
    - Can I see how the model performs?
"""