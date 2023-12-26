import math
import sympy as s
import matplotlib.pyplot as plt
import numpy as n
from matplotlib.animation import FuncAnimation

# Временная буквенная переменная
t = s.Symbol('t')
# Законы движения в декартовых координатах
x = (2 + s.sin(8 * t)) * s.cos(t + 0.5 * s.sin(4 * t))
y = (2 + s.sin(8 * t)) * s.sin(t + 0.5 * s.sin(4 * t))
# Вектор скорости
Vx = s.diff(x)
Vy = s.diff(y)
# Модуль вектора скорости
v = (Vx ** 2 + Vy ** 2) ** 0.5
# Модуль ускорения
Ax = s.diff(Vx)
Ay = s.diff(Vy)
# Модуль вектора скорости
a = (Ax ** 2 + Ay ** 2) ** 0.5
# Модуль тангенциального ускорения
Atan = s.diff(v)
# Модуль центростремительного ускорения
Acs = (a ** 2 - Atan ** 2) ** 0.5
# Тангенциальное ускорение
ATanx = Vx / v * Atan
ATany = Vy / v * Atan
# Нормальное ускорение
ANorx = Ax - ATanx
ANory = Ay - ATany
# Модуль нормального ускорения
ANor = (ANorx ** 2 + ANory ** 2) ** 0.5
# Радиус кривизны
CRX = (ANorx / ANor) * v ** 2 / Acs
CRY = (ANory / ANor) * v ** 2 / Acs


step = 2500
T = n.linspace(0, 10, step)
X = n.zeros_like(T)
Y = n.zeros_like(T)
VX = n.zeros_like(T)
VY = n.zeros_like(T)
AX = n.zeros_like(T)
AY = n.zeros_like(T)
CRx = n.zeros_like(T)
CRy = n.zeros_like(T)


for i in n.arange(len(T)):
    X[i] = s.Subs(x, t, T[i])
    Y[i] = s.Subs(y, t, T[i])
    VX[i] = s.Subs(Vx, t, T[i])
    VY[i] = s.Subs(Vy, t, T[i])
    AX[i] = s.Subs(Ax, t, T[i])
    AY[i] = s.Subs(Ay, t, T[i])
    CRx[i] = s.Subs(CRX, t, T[i])
    CRy[i] = s.Subs(CRY, t, T[i])

fig = plt.figure()

axis = fig.add_subplot(1, 1, 1)
axis.axis('equal')
axis.set(xlim = [-5, 5], ylim = [-5, 5])
axis.plot(X, Y)
Pnt = axis.plot(X[0],Y[0], marker = 'o')[0]
Vp = axis.plot([X[0], X[0] + VX[0]], [Y[0], Y[0] + VY[0]], 'r')[0]
Ap = axis.plot([X[0], X[0] + AX[0]], [Y[0], Y[0] + AY[0]], 'g')[0]
Cr = axis.plot([X[0], X[0] + CRx[0]], [Y[0], Y[0] + CRy[0]], 'black')[0]

def Vect_arrow(X, Y, ValX, ValY):
    a = 0.2
    b = 0.3
    Arx = n.array([-b, 0, -b])
    Ary = n.array([a, 0, -a])
    alp = math.atan2(ValY, ValX)
    RotArx = Arx * n.cos(alp) - Ary * n.sin(alp)
    RotAry = Arx * n.sin(alp) + Ary * n.cos(alp)

    Arx = X + ValX + RotArx
    Ary = Y + ValY + RotAry
    return Arx, Ary

AVx, AVy = Vect_arrow(X[0], Y[0], VX[0], VY[0])
AAx, AAy = Vect_arrow(X[0], Y[0], AX[0], AY[0])
ACRx, ACRy = Vect_arrow(X[0], Y[0], CRx[0], CRy[0])
Varrow = axis.plot(AVx, AVy, 'red')[0]
Aarrow = axis.plot(AAx, AAy, 'green')[0]
CRarrow = axis.plot(ACRx, ACRy, 'black')[0]

def anim(i):
    Pnt.set_data([X[i]], [Y[i]])
    Vp.set_data([X[i], X[i] + VX[i]], [Y[i], Y[i] + VY[i]])
    Ap.set_data([X[i], X[i] + AX[i]], [Y[i], Y[i] + AY[i]])
    Cr.set_data([X[i], X[i] + CRx[i]], [Y[i], Y[i] + CRy[i]])
    AVx, AVy = Vect_arrow(X[i], Y[i], VX[i], VY[i])
    AAx, AAy = Vect_arrow(X[i], Y[i], AX[i], AY[i])
    ACRx, ACRy = Vect_arrow(X[i], Y[i], CRx[i], CRy[i])
    Varrow.set_data(AVx, AVy)
    Aarrow.set_data(AAx, AAy)
    CRarrow.set_data(ACRx, ACRy)

an = FuncAnimation(fig, anim, frames = step, interval = 1)

plt.show()