import numpy 
import pandas 
import  tensorflow
from tensorflow import keras
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()
# x_train.shape
# x_test.shape
# print(y_teat)
plt.imshow(x_test[1])
plt.show() 
x_train=x_train/255
x_test=x_test/255
# print(x_train[0])
model= Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history=model.fit(x_train,y_train,epochs=10,validation_split=0.2)
print(history.history)
plt.plot(history.history['loss'],)
plt.plot(history.history['val_loss'])
plt.show()

model.save("digit.h5")
model2=load_model('digit.h5')
ypro= model2.predict(x_test)
print(ypro.argmax(axis=1))
from sklearn.metrics import accuracy_score
print(accuracy_score(ypro.argmax(axis=1),y_test))


