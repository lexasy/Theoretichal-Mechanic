import math
import sympy as s
import matplotlib.pyplot as plt
import numpy as n
from matplotlib.animation import FuncAnimation

t = s.Symbol('t')

x = (2 + s.sin(8 * t)) * s.cos(t + 0.5 * s.sin(4 * t))
y = (2 + s.sin(8 * t)) * s.sin(t + 0.5 * s.sin(4 * t))

Vx = s.diff(x)
Vy = s.diff(y)

Ax = s.diff(Vx)
Ay = s.diff(Vy)

step = 2500
T = n.linspace(0,10,step)
X = n.zeros_like(T)
Y = n.zeros_like(T)
VX = n.zeros_like(T)
VY = n.zeros_like(T)
AX = n.zeros_like(T)
AY = n.zeros_like(T)

for i in n.arange(len(T)):
    X[i] = s.Subs(x, t, T[i])
    Y[i] = s.Subs(y, t, T[i])
    VX[i] = s.Subs(Vx, t, T[i])
    VY[i] = s.Subs(Vy, t, T[i])
    AX[i] = s.Subs(Ax, t, T[i])
    AY[i] = s.Subs(Ay, t, T[i])
fig = plt.figure()

axis = fig.add_subplot(1, 1, 1)
axis.axis('equal')
axis.set(xlim = [-100, 100], ylim = [-100, 100])
axis.plot(X, Y)
Pnt = axis.plot(X[0],Y[0], marker = 'o')[0]
Vp = axis.plot([X[0], X[0] + VX[0]], [Y[0], Y[0] + VY[0]], 'r')[0]
Ap = axis.plot([X[0], X[0] + AX[0]], [Y[0], Y[0] + AY[0]], 'g')[0]

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

RAx, RAy = Vect_arrow(X[0], Y[0], VX[0], VY[0])
AAx, AAy = Vect_arrow(X[0], Y[0], AX[0], AY[0])
Varrow = axis.plot(RAx, RAy, 'red')[0]
Aarrow = axis.plot(AAx, AAy, 'green')[0]

def anim(i):
    Pnt.set_data([X[i]], [Y[i]])
    Vp.set_data([X[i], X[i] + VX[i]], [Y[i], Y[i] + VY[i]])
    Ap.set_data([X[i], X[i] + AX[i]], [Y[i], Y[i] + AY[i]])
    RAx, RAy = Vect_arrow(X[i], Y[i], VX[i], VY[i])
    AAx, AAy = Vect_arrow(X[i], Y[i], AX[i], AY[i])
    Varrow.set_data(RAx, RAy)
    Aarrow.set_data(AAx, AAy)

an = FuncAnimation(fig, anim, frames = step, interval = 1)

plt.show()