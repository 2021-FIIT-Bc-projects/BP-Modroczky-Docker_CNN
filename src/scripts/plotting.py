import numpy as np
import matplotlib.pyplot as plt
import os

path = '../plots'

x = np.linspace(start=-10, stop=10, num=1000)
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)
relu = np.maximum(0, x)

functions = [sigmoid, tanh, relu]
names = ['Sigmoid', 'tanh', 'ReLU']

for func, name in zip(functions, names):
    plt.plot(func)
    plt.legend([name + ' function'])
    plt.savefig(os.path.join(path, name + '.png'))
    plt.clf()
