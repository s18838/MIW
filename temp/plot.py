# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import pandas as pd


x = list(range(0,10))
y = list(range(0,10))
plt.figure(figsize=(7,5))
plt.plot(x, y, '--or')
plt.title("Simple example")
plt.xlabel("Element index")
plt.ylabel("Element value")
plt.show()