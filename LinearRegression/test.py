import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv("Salary_Data[1].csv")
data = data.dropna(subset=["Years_of_Experience","Salary"])

def loss_function(m,b,points):
    Y = points.Salary
    X = points.Years_of_Experience
    n = len(points)
    return np.sum((np.array(Y)-m*np.array(X)-b)**2)/n


m = 0
b = 0
L = 0.001
epochs = 1000

def gradiant_descent(m_now,b_now,L,points):
    m_gradiant = 0
    b_gradiant = 0

    n = len(points)

    Y = points.Salary
    X = points.Years_of_Experience

    b_gradiant = -2/n * np.sum(np.array(Y)-m_now*np.array(X)-b_now)
    m_gradiant = -2/n * np.sum(np.dot(np.array(X),(np.array(Y)-m_now*np.array(X)-b_now)))

    m = m_now-L*m_gradiant
    b = b_now-L*b_gradiant

    return m,b

for i in range(epochs):
    if i%50 == 0:
        print(f"Epochs {i} , m : {m} , b : {b} , loss : {loss_function(m,b,data)}")
    m,b = gradiant_descent(m,b,L,data)

print(f"m : {m:.4f} , b : {b:.4f}")


plt.scatter(data.Years_of_Experience,data.Salary)
plt.plot(list(range(0,30)),[m*x+b for x in range(0,30)],color="red")
plt.show()



