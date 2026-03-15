import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
import tensorflow 
from tensorflow import keras# through which we use all this
from tensorflow.keras.models import Sequential# for siquential model we use it 
from tensorflow.keras.layers import Dense# for creating the dense and hidden layer we use this 
from tensorflow.keras.models import load_model# for loading the save model we use this  
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
data= pd.read_csv(r"C:\Users\gsrik\Desktop\archive (6)\Bank Customer Churn Prediction.csv")
# print(data.head())
# print(data.info())
# print(data.duplicated().sum())
# print(data['churn'].value_counts())
# print(data['country'].value_counts())
# print(data['gender'].value_counts())
# print(data['age'].value_counts())
data.drop(columns=['customer_id',],inplace=True)
print(data.head())
data = pd.get_dummies(data, columns=['country','gender'],drop_first=True)# it remove the dummies data and find the required data only 
print(data.head())
from sklearn.model_selection import train_test_split
x= data.drop(columns=['churn'])
y=data['churn']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
from sklearn.preprocessing import StandardScaler
scale= StandardScaler()
x_train_scale=scale.fit_transform(x_train)
x_test_scale = scale.fit_transform(x_test)
print(x_train_scale)   
model.Sequential()
model.add(Dense(17,activation='relu',input_dim=11))
model.add(Dense(5,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])
history=model.fit(x_train_scale,y_train,epochs=10,validation_split=0.2)# it store the data in this dictoneary in the history vriable  
print(model.layers[0].get_weights())
model.save("churn_model.h5")
model1= load_model("churn_model.h5")
y_prid=model1.predict(x_test_scale)
y_prid =np.where(y_prid>=0.5,1,0)
print(y_prid)
from sklearn.metrics import  accuracy_score
acc=accuracy_score(y_test,y_prid)
print(acc)
print(np.unique(y_prid, return_counts=True))
print(history.history)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

