import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import numpy as np
import matplotlib as plt

import tensorflow_datasets as tfds


(ds_train,ds_test),ds_info = tfds.load(
    'mnist',
    split=['train','test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

def normalize_img(image,label):
    return tf.cast(image,tf.float32) / 255., label


ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE
)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

model = tf.keras.models.Sequential ([
    tf.keras.Input((28,28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#print(loss_fn(y_train[:1],predictions).numpy())

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss=loss_fn,metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
model.fit(ds_train,epochs=6,validation_data=ds_test)

#model.evaluate(ds_test, verbose=2)
