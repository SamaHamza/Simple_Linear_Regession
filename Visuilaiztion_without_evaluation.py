import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#read data
data = np.genfromtxt("fire_and_theft.txt", names=('fire','theft'), dtype=np.float64, skip_header=1)
x_values = [[x[0]] for x in data]
y_values = [[x[1]] for x in data]

#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

#visualize results
plt.title('Concentration During The Time')
plt.xlabel('Time(T)')
plt.ylabel('Dye Concentration(D)')
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values), color='green')
plt.show()


