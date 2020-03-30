# Example adapté de https://matplotlib.org/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py
import numpy as np
from matplotlib import pyplot as plt
x = range(100)
y = [i for i in x]
y_2 = [i**2 for i in x]
y_3 = [i**3 for i in x]

print(x)

plt.plot(x, y, label='linear')
plt.plot(x, y_2, label='quadratic')
plt.plot(x, y_3, label='cubic')

plt.xlabel('x label')
plt.ylabel('y label')
plt.title("Simple Plot")
plt.legend()
plt.show()