import pandas as pd
import  matplotlib.pyplot as plt

data = pd.read_csv('Salary_Data[1].csv')
data = data.dropna(subset=["Years_of_Experience","Salary"])

def loss_function(m,b,points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].Years_of_Experience
        y = points.iloc[i].Salary
        total_error += (y-(m*x + b)) ** 2
    return total_error / float(len(points))

def gradiant_descent(m_now,b_now,points,L):
    m_gradiant = 0
    b_gradiant = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].Years_of_Experience
        y = points.iloc[i].Salary
        m_gradiant += -(2/n) * x * (y-(m_now*x + b_now))
        b_gradiant += -(2 / n) * (y - (m_now * x + b_now))

    m = m_now - m_gradiant*L
    b = b_now - b_gradiant * L
    return m,b


m = 0
b = 0
L = 0.001
epochs = 1000

for i in range(epochs):
    if i%50 == 0:
        print(f"Epochs : {i} , m : {m:.4f} , b : {b:.4f} , loss : {loss_function(m,b,data)}")
    m,b = gradiant_descent(m,b,data,L)

print(m,b)
plt.scatter(data.Years_of_Experience,data.Salary,color = "black")
plt.plot(list(range(0,30)) , [m*x+b for x in range(0,30)],color="red")
plt.show()