import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses


"""
This tutorial uses a dataset consisting of 50 000 movie reviews from IMDB. 
The model will resolve if a review is positive or negative.
"""

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file("aclImdb_v1", url, untar=True, cache_dir='.',cache_subdir='') # cache_dir is the directory to cache the dataset, . means 'here'?
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')


os.listdir(dataset_dir)






