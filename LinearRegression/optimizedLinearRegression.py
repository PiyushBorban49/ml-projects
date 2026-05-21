import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy.vector import gradient
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Salary_Data[1].csv")
data = data.dropna(subset=["Years_of_Experience","Salary"])

# print(len(data))

def loss_function(m,b,points):
    return np.sum((np.array(list(points.Salary))-(m*np.array(list(points.Years_of_Experience)) + b))**2)/len(points)

# print(loss_function(len(p)))


def gradient_descent(m_now,b_now,points,L):
    m_gradiant = 0
    b_gradiant = 0
    n = len(points)
    b_gradiant = -2/n * np.sum(np.array(points.Salary)-(m_now*np.array(points.Years_of_Experience)+b_now))
    m_gradiant = -2/n * np.sum(np.dot((np.array(points.Salary) - (m_now * np.array(points.Years_of_Experience) + b_now)),np.array(points.Years_of_Experience)))
    m = m_now-(m_gradiant*L)
    b = b_now-(b_gradiant*L)
    return m,b

m = 0
b = 0
L = 0.001
epochs = 1000



for i in range(epochs):
    if i%50 == 0:
        print(f"Epochs : {i} , m : {m:.4f} , b : {b:.4f} , loss : {loss_function(m,b,data)}")
    m,b = gradient_descent(m,b,data,L)

print(f"m : {m:.4f} , b : {b:.4f}")

plt.scatter(data.Years_of_Experience,data.Salary)
plt.plot(list(range(0,30)),[m*x+b for x in range(0,30)],color="red")
plt.show()

