import numpy as np
import pandas as pd

data = pd.read_csv("iris_dataset.csv")
# data = data.dropna(subset=["sepal_length_(cm)","sepal_width_(cm)","petal_length_(cm)","petal_width_(cm)","target"])


print(data)