import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

train_data = np.loadtxt('C:/Users/Bingtao LI/PycharmProjects/text/train_data.csv', delimiter=",", skiprows=0)
train_data_mm = min_max_scaler.fit_transform(train_data)
train_data_scaled = preprocessing.scale(train_data_mm)
train_target = np.loadtxt('C:/Users/Bingtao LI/PycharmProjects/text/train_target.csv', delimiter=",", skiprows=0)
train_target.reshape(1,-1)

test_data= np.loadtxt('C:/Users/Bingtao LI/PycharmProjects/text/test_data.csv', delimiter=",", skiprows=0)
test_data_mm = min_max_scaler.fit_transform(test_data)
test_data_scaled = preprocessing.scale(test_data_mm)
test_target= np.loadtxt('C:/Users/Bingtao LI/PycharmProjects/text/test_target.csv', delimiter=",", skiprows=0)
test_target.reshape(1,-1)

lala = svm.SVC(kernel='poly',degree=2, decision_function_shape='ovr',C=1.8)
lala.fit(train_data_scaled, train_target)
z = lala.predict(test_data_scaled)

print(z)
print(test_target)
print(accuracy_score(test_target, z))