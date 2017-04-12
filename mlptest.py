import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

train_data = np.loadtxt('C:/Users/Bingtao LI/PycharmProjects/text/train_data.csv', delimiter=",", skiprows=0)
scaler.fit(train_data)
train_data_scaled = scaler.transform(train_data)
train_target = np.loadtxt('C:/Users/Bingtao LI/PycharmProjects/text/train_target.csv', delimiter=",", skiprows=0)
train_target.reshape(1,-1)

test_data= np.loadtxt('C:/Users/Bingtao LI/PycharmProjects/text/test_data.csv', delimiter=",", skiprows=0)
test_data_scaled = scaler.transform(test_data)
test_target= np.loadtxt('C:/Users/Bingtao LI/PycharmProjects/text/test_target.csv', delimiter=",", skiprows=0)
test_target.reshape(1,-1)

nn = MLPClassifier(solver='adam', hidden_layer_sizes=(20,20), random_state=1)
nn.fit(train_data_scaled, train_target)
z = nn.predict(test_data_scaled)

print(z)
print(test_target)
print(accuracy_score(test_target, z))