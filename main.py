import np as np
import pandas as pd
import numpy as np
import sklearn.model_selection
from sklearn import linear_model
from sklearn.utils import shuffle

df = pd.read_csv("weather_final.csv", sep=",")
print(df.head())

df = df[['TempAvgF', 'HumidityAvgPercent', 'SeaLevelPressureAvgInches', 'VisibilityAvgMiles', 'WindAvgMPH']]

predict = "TempAvgF"

X = np.array(df.drop([predict], 1))
Y = np.array(df[predict])

x_train, x_test, y_train, y_test, = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)
