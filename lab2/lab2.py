import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Массив времени
Steps = 1500
t = np.linspace(0, 20, Steps)

# Углы
phi = (2 + np.sin(8 * t)) * np.cos(t + 0.5 * np.sin(4 * t))
tet = (2 + np.sin(8 * t)) * np.sin(t + 0.5 * np.sin(4 * t))

# Начальные координаты точек системы
X_O = 0
Y_O = 0
X_A = np.cos(phi)
Y_A = np.sin(phi)
X_B = -np.sin(tet)
Y_B = -np.cos(tet)

# Оси
X_Ground = [0, 0, 4]
Y_Ground = [4, 0, 0]

# Пружина
Nv = 3
R1 = 0.1
R2 = 0.3
numpoints = np.linspace(0, 1, 20 * Nv + 1)
Betas = numpoints * (2 * np.pi * Nv + phi[0])
X_Spiral = np.cos(Betas) * (R1 + (R2 - R1) * numpoints)
Y_Spiral = np.sin(Betas) * (R1 + (R2 - R1) * numpoints)

# Создание области
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.axis('equal')
ax.set(xlim = [-5, 5], ylim = [-5, 5])

# Отрисовка системы координат
ax.plot(X_Ground, Y_Ground, color = 'black', linewidth = 1)

# Отрисовка стержня OA
OA = ax.plot([X_O, X_A[0]], [Y_O, Y_A[0]])[0]

# Отрисовка стержня AB
AB = ax.plot([X_A[0], X_B[0]], [Y_A[0], Y_B[0]])[0]

# Отрисовка точек О, A, B
Point_O = ax.plot(X_O, Y_O, marker = 'o')[0]
Point_A = ax.plot(X_A[0], Y_A[0], marker = 'o', markersize = 6)[0]
Point_B = ax.plot(X_B[0], Y_B[0], marker = 'o', markersize = 15)[0]

# Отрисовка пружины 
Spiral = ax.plot(X_Spiral, Y_Spiral)[0]

def run(i):
    OA.set_data([X_O, X_A[i]], [Y_O, Y_A[i]])
    AB.set_data([X_A[i], X_B[i]], [Y_A[i], Y_B[i]])
    
    Point_A.set_data(X_A[i], Y_A[i])
    Point_B.set_data(X_B[i], Y_B[i])
    
    Betas = numpoints * (2 * np.pi * Nv + phi[i])
    X_Spiral = np.cos(Betas) * (R1 + (R2 - R1) * numpoints)
    Y_Spiral = np.sin(Betas) * (R1 + (R2 - R1) * numpoints)
    Spiral.set_data(X_Spiral, Y_Spiral)
    
    return OA, AB, Point_A, Point_B, Betas, X_Spiral, Y_Spiral, Spiral

anim = FuncAnimation(fig, run, frames = len(t), interval = 50, repeat = False)

plt.show()