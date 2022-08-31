import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.python.framework.type_spec import ops
from tensorflow.python.ops import math_ops
import datetime
import PrepareDataset
from sklearn.model_selection import train_test_split

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
#from tensorflow.keras import callbacks

import winsound

#import tensorflow.keras.losses as kl




print(tf.__version__)
#%%
x,y = PrepareDataset.load_dataset('newest')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
#x_train, y_train = x, y
#x_train = y
#%%
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(x_train)

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=400,
  decay_rate=0.5,
  staircase=True)
early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=200,
    verbose=1,
    mode='auto',
    #baseline=None,
    restore_best_weights=True
)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#%%
print(normalizer.mean.numpy())
first = np.array(x_train[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())

#%%


model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=256, input_shape=(x_train.shape[1],), activation='relu'),
    #layers.Dropout(rate = 0.05),
    #layers.Dense(units=2048, activation='relu'),
    #layers.Dense(units=128, activation='relu'), ##
    #layers.Dense(units=128, activation='relu'),
    layers.Dense(units=128, activation='relu'),
    #layers.Dropout(rate = 0.05),
    layers.Dense(units=128, activation='relu'),
    #layers.Dropout(rate = 0.05),
    layers.Dense(units=32, activation='relu'),
    layers.Dense(units=4)
])

model.summary()
#%%
print(y_train[:10])
print(model.predict(x_train[:10]))

yt = y_train[:10]
yp = model.predict(x_train[:10])

#%%
def custom_loss(y_true,y_pred):
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)

    y_diff = tf.subtract(y_pred, y_true)

    only_pos = y_diff[:, 0:2] # tf.slice(y_diff, begin=0, size=2) #
    only_blink = y_diff[:, 2:4] # tf.slice(y_diff, begin=[2], size=[2]) #

    only_pos_sqr = tf.multiply(only_pos, only_pos)
    only_blink_abs = tf.abs(only_blink)

    blink_sum = tf.reduce_sum(only_blink_abs, -1) # * 2
    pos_sum = tf.reduce_sum(only_pos_sqr,-1)

    geom_err = tf.pow(pos_sum, tf.constant([0.5]))
    blink_err = tf.exp(blink_sum)
    return tf.multiply(geom_err, blink_err)
def geom_err(y_true,y_pred):
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)

    y_diff = tf.subtract(y_pred, y_true)

    only_pos = y_diff[:, 0:2] # tf.slice(y_diff, begin=0, size=2) #
    only_blink = y_diff[:, 2:4] # tf.slice(y_diff, begin=[2], size=[2]) #

    only_pos_sqr = tf.multiply(only_pos, only_pos)
    only_blink_abs = tf.abs(only_blink)

    blink_sum = tf.reduce_sum(only_blink_abs, -1) # * 2
    pos_sum = tf.reduce_sum(only_pos_sqr,-1)

    geom_err = tf.pow(pos_sum, tf.constant([0.5]))
    #blink_err = tf.exp(blink_sum)
    #return tf.multiply(geom_err, blink_err)
    return geom_err
#custom_loss(yt,yp)
#%%

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate = lr_schedule),
    loss=custom_loss,
    metrics=[geom_err]
)

#%%time
# history_first10 = model.fit(
#     x_train,
#     y_train,
#     epochs=10,
#     # Suppress logging.
#     #verbose=1,
#     # Calculate validation results on 20% of the training data.
#     validation_split = 0.2)
#%%
history = model.fit(
    x_train,
    y_train,
    epochs=10000,
    # Suppress logging.
    #verbose=1,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2,
    callbacks=[early_stop_callback,
               tensorboard_callback]
)
#%%
def plot_loss(history):
  loss1 =  np.convolve(history.history['geom_err'], np.ones(5), 'valid') / 5
  loss2 = np.convolve(history.history['val_geom_err'], np.ones(5), 'valid') / 5

  plt.plot(loss1, label='geom_err')
  plt.plot(loss2, label='val_geom_err')
  #plt.plot(history.history['loss'], label='loss')
  #plt.plot(history.history['val_loss'], label='val_loss')

  #plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)
  plt.show()
plot_loss(history)

predictions = model.predict(x_test)
diff = predictions - y_test
err = np.sqrt(diff[:,0]**2 + diff[:,1]**2)

print(f'Median error = {np.median(err)} pixels, which is {np.median(err)/1400*100:.2f}% error')
print(f'Mean error = {np.mean(err)} pixels, which is {np.mean(err)/1400*100:.2f}% error')

model.save('Models/main')
winsound.Beep(500, 700)

pass