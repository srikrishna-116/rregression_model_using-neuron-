import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
data= pd.read_csv(r"C:\Users\gsrik\Desktop\archive (6)\Admission_Predict.csv")
print(data.head())
print(data.info())
print(data.duplicated)
data.drop(columns=['Serial No.'],inplace=True)
x=data.iloc[:,0:-1]# from 0 index not include -1
y=data.iloc[:,-1]
print(x.head())
print(y.head())
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
from sklearn .preprocessing import MinMaxScaler
scaler= MinMaxScaler()
x_train_scale=scaler.fit_transform(x_train)
x_test_scale=scaler.fit_transform(x_test)
model=Sequential()
model.add(Dense(7,activation='relu',input_dim=7))
model.add(Dense(7,activation='relu'))
model.add(Dense(1,activation='linear'))#when we are doing regression we use linear as output activation function.
model.summary()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
history=model.fit(x_train_scale,y_train,epochs=100,validation_split=0.2)
print(history.history)
plt.plot(history.history['loss'],)
plt.plot(history.history['val_loss'])
plt.show()
model.save('regression.h5')
y_pre=model.predict(x_test_scale)
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pre))
