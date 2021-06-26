import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datos = pd.read_csv("datos.csv", sep=' ')
datos = datos.to_numpy()

b = 0
W = np.zeros([1, 1])

alpha = 0.000001
x1 = datos[:, 0]
Y = datos[:, 1]
m = x1.size

x1 = x1.reshape([x1.size, 1])
Y = Y.reshape([Y.size, 1])


for iter1 in range(100):
    acum0 = (np.dot(x1, W) + b) - Y
    acum1 = acum0 * x1

    b -= (np.mean(acum0) * alpha)
    W -= (np.mean(acum1) * alpha)

Yp = x1 * W + b
plt.plot(x1, Y, "b-")
plt.plot(x1, Yp, "r-")

print(" PROBLEMA 1 ")
print(f"Solucion: {b}+{W}")

print("\n")


## REGRESION MULTIPLE

def escalar(x):
    xmin = np.min(x, axis=0)
    xmax = np.max(x, axis=0)
    xmean = np.mean(x, axis=0)
    xrango = xmax - xmin
    x = ( x - xmean ) / xrango
    return xmean, xrango, x

def prediccion(xmean, xrango, b, W, x):
    x = (x - xmean) / xrango
    Yp = np.dot(x, W) + b
    return Yp

## VARIABLES AJUSTADAS A LO QUE SE PIDE EN LA TAREA, NO ES GENERAL

alpha = 0.1
n = 2 ## Dos porque en este caso, segun la tarea se piden dos variables de x (x, x**2)
W = np.zeros([n, 1])
x2 = x1**2 ## Se eleva la x2 al cuadrado
x = np.concatenate((x1, x2), axis=1) ## Se concatenan a la x para tener las x1 y x2 juntas
xmean, xrango, x = escalar(x)

for iter1 in range(1000):
    acum0 = (np.dot(x, W) + b) - Y
    acum1 = acum0 * x

    acum0 = np.mean(acum0, axis=0)
    acum1 = np.mean(acum1, axis=0)

    acum1 = acum1.reshape([n, 1])

    b -= acum0 * alpha
    W -= acum1 * alpha


print(" PROBLEMA 2 ")
print(f"Solucion: \n b={b} \n + \n w={W}")
x = np.concatenate((x1, x2), axis=1)
Ym = prediccion(xmean, xrango, b, W, x)

z = np.concatenate((Y, Ym, Y-Ym), axis=1)
print(f"Z = \n {z}")

plt.xlabel("A")
plt.ylabel("Y")
plt.grid()
plt.show()

plt.plot(x2, Y, "b-")
plt.plot(x2, Ym, "r-")
plt.xlabel("A")
plt.ylabel("Y")
plt.grid()
plt.show()