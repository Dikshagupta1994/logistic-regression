from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

dataset=pd.read_csv('C:/Users\hp\Downloads\Social_Network_Ads.csv')
dataset.head()

x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,
                    test_size=0.20,random_state=0)

classifer=LogisticRegression()
classifer.fit(x_train,y_train)

y_pred=classifer.predict(x_test)
classifer.score(x_train,y_train)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)
acc=(sum(np.diag(cm))/len(y_test))
print(acc)

from sklearn import metrics
metrics.accuracy_score(y_test,y_pred)











