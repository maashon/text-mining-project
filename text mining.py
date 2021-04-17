# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:07:38 2019

@author: Parsa
"""


"""  loadin dataset and preprpcessing"""
import pandas as pd 
data = pd.read_excel('C:/Users/Parsa/Desktop/text mining/4/parsa/text mining2/public_exam_mydata.xls')
data = data[['comment','label']]
max_fatures = 2500
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['comment'].values)
X = tokenizer.texts_to_sequences(data['comment'].values)
from keras.preprocessing.sequence import pad_sequences
X = pad_sequences(X)
embed_dim = 128
lstm_out = 196
""" building model """


from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.optimizers import RMSprop
model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3,activation='softmax'))
model.compile(optimizer=RMSprop(),loss='categorical_crossentropy',metrics=['accuracy'])
#model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

""" dividing dataset to test and train set """
from sklearn.model_selection import train_test_split
batch_size = 128
validation_size = 30
Y = pd.get_dummies(data['label']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]

temp = model.fit(X_train, Y_train, epochs = 12, batch_size=batch_size,shuffle=True, verbose = 1,validation_data=(X_validate,Y_validate))
#hh=model.evaluate(test_data,test_label)
resault=temp.history
accuracy=resault['acc']
loss=resault['loss']
v_accuracy=resault['val_acc']
v_loss=resault['val_loss']



""" plotting the model hsitory"""

import matplotlib.pyplot as plt
ep=range(1,len(accuracy)+1)
plt.plot(ep,accuracy,'b',label='accuracy')
plt.plot(ep,loss,'r',label='loss')
plt.plot(ep,v_accuracy,'c',label='val-accuracy')
plt.plot(ep,v_loss,'m',label='val-loss')

plt.xlabel("epoches")
plt.ylabel("acc and loss")
plt.legend()
model.save('modelRms.h5')
model.save_weights('weightsRms.h5')


#mazloomzadeh.courses@gamail.com

""" evaluating section """
from keras.models import load_model
import numpy as np
l_model = load_model('C:/Users/Parsa/Desktop/text mining/5/modelRms.h5')

X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
Y_test=np.delete(Y_test,3,axis=1)
loss,acc = l_model.evaluate(X_test, Y_test, verbose = 1, batch_size = batch_size)
print("loss: %.2f" % (loss))
print("acc: %.2f" % (acc))


#twt = ['not a good one']
##vectorizing the tweet by the pre-fitted tokenizer instance
#twt = tokenizer.texts_to_sequences(twt)
##padding the tweet to have exactly the same shape as `embedding_2` input
##twt = pad_sequences(twt, maxlen=28, dtype='int32', value=0)
#twt = pad_sequences(twt, maxlen=28, dtype='int32')
#print(twt)
#sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
#if(np.argmax(sentiment) == 0):
#    print("negative")
#elif (np.argmax(sentiment) == 1):
#    print("positive")
