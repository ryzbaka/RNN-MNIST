import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout

(trainx,trainy),(testx,testy)=tf.keras.datasets.mnist.load_data()

trainx=trainx/255.0
testx=testx/255.0

model=Sequential()

model.add(LSTM(128,input_shape=trainx.shape[1:],activation='relu',return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))

optimize=tf.keras.optimizers.Adam(lr=1e-3,decay=1e-5)

model.compile(loss='sparse_categorical_crossentropy',optimizer=optimize,metrics=['accuracy'])

model.fit(trainx,trainy,epochs=5,validation_data=(testx,testy))