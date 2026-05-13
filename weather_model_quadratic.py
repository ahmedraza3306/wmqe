import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("data/weather.csv")

x = data["Day"]
y = data["Temperature"]

# Quadratic regression
coefficients = np.polyfit(x, y, 2)
model = np.poly1d(coefficients)

# Prediction
x_line = np.linspace(min(x), max(x), 100)
y_line = model(x_line)

# Plotting
plt.scatter(x, y)
plt.plot(x_line, y_line)
plt.xlabel("Day")
plt.ylabel("Temperature")
plt.title("Weather Modelling using Quadratic Regression")
plt.show()
