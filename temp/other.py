import numpy as np


x = [1, 2, 3]
input_size = len(x)
layer_size = 5

weight = np.random.rand(layer_size, input_size) * 2 - 1
bias = np.random.rand(layer_size) * (-1)

print(weight)
print(bias)

u = np.matmul(weight, x)
print(u)
u_prim = u + bias
print(u_prim)
y = 1 / (1 + np.exp(-u_prim))

print(y)

print(np.argmax(y))

x = np.arange(0, 100, 1)

print(x)


X = x.reshape(100, 1)



print(X)

ones = np.ones((x.shape[0], 1))
               
print(ones)

X_ = np.c_[ones, X]

print(X_)


x = np.array([1,2,3])

print(x)
print(x.T)

x = np.atleast_2d(x)


print(x)
print(x.T)