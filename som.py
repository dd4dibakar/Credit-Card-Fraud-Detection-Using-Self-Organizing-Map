# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data = pd.read_csv('Credit_Dataset.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler(feature_range = (0, 1))
X = scale.fit_transform(X)

# Training the SOM
from minisom import MiniSom
s_o_m = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
s_o_m.random_weights_init(X)
s_o_m.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(s_o_m.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = s_o_m.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = s_o_m.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
frauds = scale.inverse_transform(frauds)