import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import neighbors
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

train_data = np.loadtxt('C:/Users/Bingtao LI/PycharmProjects/text/train_data.csv', delimiter=",", skiprows=0)
train_target = np.loadtxt('C:/Users/Bingtao LI/PycharmProjects/text/train_target.csv', delimiter=",", skiprows=0)
train_target.reshape(1,-1)
test_data= np.loadtxt('C:/Users/Bingtao LI/PycharmProjects/text/test_data.csv', delimiter=",", skiprows=0)
test_target= np.loadtxt('C:/Users/Bingtao LI/PycharmProjects/text/test_target.csv', delimiter=",", skiprows=0)
test_target.reshape(1,-1)

knn = neighbors.KNeighborsClassifier(5)
knn.fit(train_data,train_target)


z = knn.predict(test_data)

print(accuracy_score(test_target, z))







