import numpy as np
import sympy as s
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

# Решение системы дифференциальных уравнений второго порядка
def odesys(y, t, g, l, c, k1, k2, m):
    dy = np.zeros_like(y)
    dy[0] = y[2]
    dy[1] = y[3]
    a11 = 1
    a12 = -np.cos(y[0] - y[1])
    a21 = -np.cos(y[0] - y[1])
    a22 = 1
    b1 = (g / l) * np.sin(y[0]) - (c * y[0] + k1 * y[2]) / (m * l ** 2) + y[3] ** 2 * np.sin(y[0] - y[1])
    b2 = -(g / l) * np.sin(y[1]) - k2 * y[3] / (m * l ** 2) - y[2] ** 2 * np.sin(y[0] - y[1])
    dy[2] = (b1 * a22 - b2 * a12) / (a11 * a22 - a12 * a21)
    dy[3] = (b2 * a11 - b1 * a21) / (a11 * a22 - a12 * a21)
    return dy

# Входные данные
m = 0.3
l = 1	
c = 2
k1 = 1
k2 = 1
g = 9.81
l_OA = l
l_AB = l
phi0 = np.pi / 3
tet0 = -np.pi / 3
dphi0 = 0
dtet0 = 0
# Начальное условие
y0 = [phi0, tet0, dphi0, dtet0]
# Массив времени
Steps = 1500
t = np.linspace(0, 20, Steps)
# Интегрирование
Y = odeint(odesys, y0, t, (g, l, c, k1, k2, m))

# Значения углов и их производных
phi = Y[:, 0]
tet = Y[:, 1]
dphi = Y[:, 2]
dtet = Y[:, 3]

# Фигура с графиками реакций
figstat = plt.figure()

plphi = figstat.add_subplot(4, 1, 1)
plphi.plot(t, phi)

pltet = figstat.add_subplot(4, 1, 2)
pltet.plot(t, tet)

ddphi = np.zeros_like(t)
ddtet = np.zeros_like(t)
# Получаем вторые производные углов
for i in range(len(t)):
    ddphi[i] = odesys(Y[i], t[i], g, l, c, k1, k2, m)[2]
    ddtet[i] = odesys(Y[i], t[i], g, l, c, k1, k2, m)[3]

Rx = m * l * (ddtet * np.cos(tet) - dtet ** 2 * np.sin(tet) - ddphi * np.cos(phi) + dphi ** 2 * np.sin(phi))
Ry = m * l * (ddtet * np.sin(tet) + dtet ** 2 * np.cos(tet) - ddphi * np.sin(phi) - dphi ** 2 * np.cos(phi)) + m * g

pRx = figstat.add_subplot(4, 1, 3)
pRx.plot(t, Rx)

pRy = figstat.add_subplot(4, 1, 4)
pRy.plot(t, Ry)

# Начальные координаты точек системы
X_O = 0
Y_O = 0
X_A = l_OA * np.cos(phi)
Y_A = l_OA * np.sin(phi)
X_B = l_AB * (-np.sin(tet))
Y_B = l_AB * (-np.cos(tet))

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