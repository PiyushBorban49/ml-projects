import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('Housing.csv')
# data = data.dropna(subset=['price','area','bedrooms','bathrooms','stories','mainroad','guestroom','basement','hotwaterheating','airconditioning','parking','prefarea','furnishingstatus'])

available_features = ['area','bedrooms','bathrooms','stories','parking']


X = data[available_features].values
scaler = RobustScaler()
scaler.fit(X)
X = scaler.transform(X)

Y = data['price'].values



m = np.random.normal(0,0.01,X.shape[1])
b = 0
learning_rate = 0.1
epochs = 1000

def entropy_loss(m,b,x,y):
    n = len(y)
    return (np.sum((y-(np.dot(x,m)+b))**2)/2*n)

def gradiant_descent(m_now,b_now,x,y,learning_rate):
    n = len(y)
    b_gradiant = -np.sum(y-(np.dot(x,m_now)+b_now))/n
    m_gradiant = -np.dot(x.T,(y-(np.dot(x,m_now)+b_now)))/n
    m_now -= m_gradiant*learning_rate
    b_now -= b_gradiant*learning_rate
    return m_now,b_now

for i in range(epochs):
    if i%50 == 0:
        print(f"Loss:{entropy_loss(m,b,X,Y)}")
    m,b = gradiant_descent(m,b,X,Y,learning_rate)



# Final evaluation
final_loss = entropy_loss(m,b,X,Y)
predictions = np.dot(X,m) + b
r2_score = 1 - (np.sum((Y - predictions)**2) / np.sum((Y - np.mean(Y))**2))

print(f"\nFinal Loss: {final_loss}")
print(f"R² Score: {r2_score}")
print(f"Final weights: {m}")
print(f"Final bias: {b}")

# Plot results
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(Y, predictions, alpha=0.5)
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Predictions vs Actual (R² = {r2_score:.3f})')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.hist(Y - predictions, bins=30, alpha=0.7)
plt.xlabel('Residuals (Actual - Predicted)')
plt.ylabel('Frequency')
plt.title('Residuals Distribution')
plt.grid(True)

plt.tight_layout()
plt.show()