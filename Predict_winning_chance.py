import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#reading data
train = pd.read_csv("match_history.csv")
eval = pd.read_csv("match_history_2.csv")

#swapping columns in data to create variations of it
df = pd.DataFrame([],columns=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
x = 0
y = 0
all_train = pd.DataFrame()
for x in range(0,5):
  for i in range(1,6):
    index = i+x
    if (index>5):
      index = index - 5
    df[str(index)] = train[str(i)].copy()
  for y in range(0,5):
    for a in range(6,11):
      index2 = a+y
      if (index2>10):
        index2 = index2 - 5
      df[str(index2)] = train[str(a)].copy()
    med = pd.concat((df, train["win"].copy()), axis=1)
    all_train = pd.concat((all_train, med), axis=0, ignore_index=True)

df1 = pd.DataFrame([],columns=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
x = 0
y = 0
all_eval = pd.DataFrame()
for x in range(0,5):
  for i in range(1,6):
    index = i+x
    if (index>5):
      index = index - 5
    df1[str(index)] = eval[str(i)].copy()
  for y in range(0,5):
    for a in range(6,11):
      index2 = a+y
      if (index2>10):
        index2 = index2 - 5
      df1[str(index2)] = eval[str(a)].copy()
    med = pd.concat((df1, eval["win"].copy()), axis=1)
    all_eval = pd.concat((all_eval, med), axis=0, ignore_index=True)

all_train.to_excel("train_out.xlsx", index=False)
y_all_train = all_train.pop("win")
y_all_eval = all_eval.pop("win")

x_train = all_train.to_numpy()
y_train = y_all_train.to_numpy()
x_eval = all_eval.to_numpy()
y_eval = y_all_eval.to_numpy()

dft = pd.DataFrame(x_train)
dft.to_excel("train_out.xlsx", index=False)

heros = np.unique(x_train)
#np.random.shuffle(heros)
print(heros[24])
n,d = x_train.shape
for x in range(n):
  for y in range(d):
    x_train[x][y] = np.where(heros == x_train[x][y])[0][0]

n1,d1 = x_eval.shape
for x in range(n1):
  for y in range(d1):
    x_eval[x][y] = np.where(heros == x_eval[x][y])[0][0]

x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
x_eval = (x_eval - np.min(x_eval)) / (np.max(x_eval) - np.min(x_eval))

#dft2 = pd.DataFrame(x_train)
#dft2.to_excel("out.xlsx", index=False)

model = keras.Sequential(
  [
    layers.Dense(16, input_shape=(10,), activation= "relu"),
    layers.Dense(32, activation= "relu"),
    layers.Dense(64, activation= "relu"),
    layers.Dense(128, activation= "relu"),
    layers.Dense(64, activation= "relu"),
    layers.Dense(32, activation= "relu"),
    layers.Dense(1, activation= "sigmoid")
  ]
)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
x_train = np.asarray(x_train).astype('float32')
y_train = np.asarray(y_train).astype('float32')
x_eval = np.asarray(x_eval).astype('float32')
y_eval = np.asarray(y_eval).astype('float32')
model.fit(x=x_train, y=y_train, epochs=50, batch_size=8)

#predict training dataset
y_train_pred = model.predict(x_train)
for x in range(n):
  if (y_train_pred[x] > 0.500000000000):
    y_train_pred[x] = 1
  else:
    y_train_pred[x] = 0
accuracy = np.mean(y_train == np.argmax(y_train_pred, axis=1))
print("The accuracy of training dataset is " + str(accuracy))

#predict validation dataset
y_eval_pred = model.predict(x_eval)
for x in range(n1):
  if (y_eval_pred[x] > 0.500000000000):
    y_eval_pred[x] = 1
  else:
    y_eval_pred[x] = 0
accuracy = np.mean(y_eval == np.argmax(y_eval_pred, axis=1))
print("The accuracy of validation set is " + str(accuracy))








