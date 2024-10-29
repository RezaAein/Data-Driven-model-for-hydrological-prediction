import pandas as pd
import numpy as np

df=pd.read_excel('Total.xlsx')
print(df)
print(df.isnull().sum())

df['debi']=df['debi'].fillna(df['debi'].mean())
df['tabkhir']=df['tabkhir'].fillna(df['tabkhir'].mean())

print(df.isnull().sum())

df.index=pd.to_datetime(df['Date'], format='%d.%m.%Y %H:%M:%S')
#df=df.resample('1d').mean()
df[:26]

temp2=df.iloc[:,1:]
temp2.head(10)

for i in ((temp2.columns)):
  temp2[i]=temp2[i]/max(temp2[i])

temp2.head(10)

def dfM_to_X_y(df, window_size=12):
  df_as_np=df.to_numpy()
  X=[]
  y=[]
  for i in range (len(df_as_np)-window_size):
    row=[r for r in df_as_np[i:i+window_size]]
    X.append(row)
    label=df_as_np[i+window_size][7]
    y.append(label)
  return np.array(X), np.array(y)

window_size=12
X1,y1=dfM_to_X_y(temp2, window_size)
X1.shape , y1.shape

lb=np.ceil(0.8*X1.shape[0])
ub=np.ceil(0.4*(X1.shape[0]-lb))
lb=int(lb)
ub=int(ub)

X_train1, y_train1=X1[:lb], y1[:lb]
X_val1, y_val1=X1[lb:lb+ub], y1[lb:lb+ub]
X_test1, y_test1=X1[lb+ub:], y1[lb+ub:]

X_train1.shape,y_train1.shape,X_val1.shape,y_val1.shape,X_test1.shape,y_test1.shape

from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError

model1=Sequential()
model1.add(InputLayer((X_train1.shape[1],X_train1.shape[2])))
model1.add(LSTM(128))
model1.add(Dropout(0.2))
model1.add(Dense(128,'relu'))
model1.add(Dense(128,'relu'))
model1.add(Dense(1,'linear'))
model1.summary()

cp=ModelCheckpoint('model1/', save_best_only=True)
model1.compile(loss=MeanSquaredError(), optimizer='adam', metrics=[RootMeanSquaredError()])

model1.fit(X_train1,y_train1, validation_data=(X_val1,y_val1), epochs=30, callbacks=[cp])



from keras.models import load_model
model1=load_model('model1/')

train_predictions=model1.predict(X_train1).flatten()
train_results=pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train1})
train_results

import matplotlib.pyplot as plt
plt.plot(train_results['Train Predictions'],linewidth=5)
plt.plot(train_results['Actuals'])
plt.legend(['Prediction', 'Actual'])
plt.title('Training')

val_predictions=model1.predict(X_val1).flatten()
val_results=pd.DataFrame(data={'val Predictions':val_predictions, 'Actuals':y_val1})
val_results

test_predictions=model1.predict(X_test1).flatten()
test_results=pd.DataFrame(data={'test Predictions':test_predictions, 'Actuals':y_test1})
test_results

import matplotlib.pyplot as plt
plt.plot(train_results['Train Predictions'],linewidth=5)
plt.plot(train_results['Actuals'])

plt.plot(train_results['Train Predictions'][0:1000])
plt.plot(train_results['Actuals'][0:1000])
plt.legend(['Prediction', 'Actual'])
plt.title('Training')

plt.plot(val_results['val Predictions'][0:1000])
plt.plot(val_results['Actuals'][0:1000])
plt.legend(['Prediction', 'Actual'])
plt.title('validation')

plt.plot(val_results['val Predictions'][500:1000])
plt.plot(val_results['Actuals'][500:1000])

plt.plot(test_results['test Predictions'][500:1000])
plt.plot(test_results['Actuals'][500:1000])

from sklearn.metrics import mean_squared_error as mse
def plot_predictions1(model,X,y,start=0, end=1000):
  predictions=model.predict(X).flatten()
  dff=pd.DataFrame(data={'predictions':predictions, 'Actuals':y})
  plt.plot(dff['predictions'][start:end], linewidth=2)
  plt.plot(dff['Actuals'][start:end])
  plt.legend(['Prediction', 'Actual'])
  plt.title('Test MSE= '+str(mse(y,predictions)))
  plt.rcParams['figure.figsize'] = [24, 8]
  return dff, mse(y,predictions)

plot_predictions1(model1,X_test1,y_test1)

model2=Sequential()
model2.add(InputLayer((X_train1.shape[1],X_train1.shape[2])))
model2.add(Conv1D(128, kernel_size=2))
model2.add(Flatten())
model2.add(Dense(128,'relu'))
model2.add(Dense(128,'relu'))
model2.add(Dense(1,'linear'))
model2.summary()

cp2=ModelCheckpoint('model2/', save_best_only=True)
model2.compile(loss=MeanSquaredError(), optimizer='adam', metrics=[RootMeanSquaredError()])

model2.fit(X_train1,y_train1, validation_data=(X_val1,y_val1), epochs=30, callbacks=[cp2])

plot_predictions1(model2,X_test1,y_test1)

model3=Sequential()
model3.add(InputLayer((X_train1.shape[1],X_train1.shape[2])))
model3.add(GRU(128))
model3.add(Dense(128,'relu'))
model3.add(Dense(128,'relu'))
model3.add(Dense(1,'linear'))
model3.summary()

cp3=ModelCheckpoint('model3/', save_best_only=True)
model3.compile(loss=MeanSquaredError(), optimizer='adam', metrics=[RootMeanSquaredError()])

model3.fit(X_train1,y_train1, validation_data=(X_val1,y_val1), epochs=30, callbacks=[cp3])

plot_predictions1(model3,X_test1,y_test1)

model4=Sequential()
model4.add(InputLayer((X_train1.shape[1],X_train1.shape[2])))
model4.add(Conv1D(128, kernel_size=2))
model4.add(MaxPooling1D(pool_size=2))
model4.add(LSTM(128))
model4.add(Dense(128,'relu'))
model4.add(Dense(128,'relu'))
model4.add(Dense(1,'linear'))
model4.summary()

cp4=ModelCheckpoint('model4/', save_best_only=True)
model4.compile(loss=MeanSquaredError(), optimizer='adam', metrics=[RootMeanSquaredError()])

model4.fit(X_train1,y_train1, validation_data=(X_val1,y_val1), epochs=30, callbacks=[cp4])

plot_predictions1(model4,X_test1,y_test1)

model5=Sequential()
model5.add(InputLayer((X_train1.shape[1],X_train1.shape[2])))
model5.add(Conv1D(128, kernel_size=2))
model5.add(MaxPooling1D(pool_size=2))
model5.add(GRU(128))
model5.add(Dense(128,'relu'))
model5.add(Dense(128,'relu'))
model5.add(Dense(1,'linear'))
model5.summary()

cp5=ModelCheckpoint('model5/', save_best_only=True)
model5.compile(loss=MeanSquaredError(), optimizer='adam', metrics=[RootMeanSquaredError()])

model5.fit(X_train1,y_train1, validation_data=(X_val1,y_val1), epochs=30, callbacks=[cp5])

plot_predictions1(model5,X_test1,y_test1)
