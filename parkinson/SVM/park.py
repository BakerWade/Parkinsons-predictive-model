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

from sklearn.model_selection import train_test_split

X = park.drop(['name','status'],axis=1)
y = park['status']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

params = {'C': [1,10,100,1000,10000], 'gamma': [1,0.1,0.001,0.0001]}
grid = GridSearchCV(SVC(),params, refit=True, verbose=1)

model = SVC()
grid.fit(X_train,y_train)

pred = grid.predict(X_test)

from sklearn.metrics import r2_score, classification_report, confusion_matrix

print('R2_score:',r2_score(y_test,pred))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
