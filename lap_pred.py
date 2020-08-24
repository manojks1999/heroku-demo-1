import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

dataset=pd.read_csv('laptop_pricing.csv')
dataset=np.array(dataset)


x=dataset[:, 1:-1]
y=dataset[:, -1]

x=x.astype('float')
y=y.astype('float')


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
classifier=LogisticRegression()

classifier.fit(x_train,y_train)


pickle.dump(classifier,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))