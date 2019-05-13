import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import numpy as np
import time

def train():
    df = pd.read_csv('/home/mineadmin/Machine_Learning_Test/k-NN/training.csv')
    train, test = train_test_split(df, test_size = 0.3)

    df = pd.get_dummies(df)

    x_train = train.drop('dist', axis=1)
    y_train = train['dist']

    x_test = test.drop('dist', axis=1)
    y_test = test['dist']

    scaler = MinMaxScaler(feature_range=(0,1))

    x_train_scaled = scaler.fit_transform(x_train)
    x_train = pd.DataFrame(x_train_scaled)

    x_test_scaled = scaler.fit_transform(x_test)
    x_test = pd.DataFrame(x_test_scaled)

    params = {'n_neighbors':[1,2,3,4,5]}

    knn= neighbors.KNeighborsRegressor()
    model = GridSearchCV(knn, params, cv=5)
    model.fit(x_train,y_train)
    best_k = model.best_params_

    k = int(best_k['n_neighbors'])

    return k, x_train, y_train

def predict(k, x_train, y_train):
    model = neighbors.KNeighborsRegressor(n_neighbors=k)
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    return pred

#rmse_val = []
#for k in range(20):
    #k = k+1
    #model = neighbors.KNeighborsRegressor(n_neighbors=k)

    #model.fit(x_train,y_train)
    #pred = model.predict(x_test)
    #error = sqrt(mean_squared_error(y_test,pred))
    #rmse_val.append(error)
#curve = pd.DataFrame(rmse_val)
#curve.plot()
#plt.show()

