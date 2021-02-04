import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('parkinsons.data')
park = pd.DataFrame(data=df,columns=df.columns)

df_corr = park.groupby('status').corr()

#print(df_corr)
#sns.heatmap(park.corr(),cmap='viridis',annot=True)

#sns.jointplot(x='Jitter:DDP',y='DFA', data=park, hue='status',palette='rocket_r')
plt.show()
#print(park.columns.nunique())
from sklearn.model_selection import train_test_split

X = park.drop(['name','status'],axis=1).values
y = park['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=68)

import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

model = Sequential()

model.add(Dense(44,activation='relu'))
#model.add(Dropout())

model.add(Dense(88,activation='relu'))
#model.add(Dropout())

model.add(Dense(44,activation='relu'))
#model.add(Dropout())

model.add(Dense(22,activation='relu'))
#model.add(Dropout())

model.add(Dense(11,activation='sigmoid'))
#model.add(Dropout())

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x=X_train,y=y_train,epochs=250,validation_data=(X_test,y_test),verbose=2)

model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
plt.show()

model.save('parkison.keris')

pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error,r2_score

print('R2_score:',r2_score(y_test,pred))
print('mean Squared error:', mean_squared_error(y_test,pred))