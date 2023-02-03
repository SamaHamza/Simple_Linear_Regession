import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt

#read data
data = np.genfromtxt("Dataset.txt", names=('T','D'), dtype=np.float64, skip_header=1)
x_values = [[x[0]] for x in data]
y_values = [[x[1]] for x in data]

#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)
y_pred = body_reg.predict(x_values)

#visualize results
plt.title('Concentration During The Time')
plt.xlabel('Time(T)')
plt.ylabel('Dye Concentration(D)')
plt.scatter(x_values, y_values)
plt.plot(x_values, y_pred, color='green')
plt.show()

# Model evaluation: In linear regression we evaluate the model by ERROR metrices not via accuracy scores:

mse = mean_squared_error(x_values, y_pred)  
rmse = math.sqrt(mse)  
print("Mean Squared Error:", rmse)

MSE = np.square(np.subtract(x_values, y_pred)).mean()   
rsme = math.sqrt(MSE)  
print("Root Mean Square Error: ", rsme)  
