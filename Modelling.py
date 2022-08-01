import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import PrepareDataset
from sklearn.model_selection import train_test_split

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
#from tensorflow.keras import callbacks

import winsound



print(tf.__version__)
#%%
# url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
# column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
#                 'Acceleration', 'Model Year', 'Origin']
#
# raw_dataset = pd.read_csv(url, names=column_names,
#                           na_values='?', comment='\t',
#                           sep=' ', skipinitialspace=True)

x,y = PrepareDataset.load_dataset('newest')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
#x_train, y_train = x, y
#x_train = y
#%%
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(x_train)

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.005,
  decay_steps=56*100,
  decay_rate=1,
  staircase=False)


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
    layers.Dense(units=2)
])

model.summary()
#%%
print(y_train[:10])
print(model.predict(x_train[:10]))

#%%
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.007),
    loss='mean_squared_error')

#%%time
history_first10 = model.fit(
    x_train,
    y_train,
    epochs=10,
    # Suppress logging.
    #verbose=1,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.1)
#%%
history = model.fit(
    x_train,
    y_train,
    epochs=500,
    # Suppress logging.
    #verbose=1,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.1,
    #callbacks=get_callbacks("lol")
)
#%%
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
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

print(f'Median error = {np.median(err)} pixels, which is {np.median(err)/1400*100}% error')
print(f'Mean error = {np.mean(err)} pixels, which is {np.mean(err)/1400*100}% error')

model.save('Models/main')
winsound.Beep(500, 700)

pass