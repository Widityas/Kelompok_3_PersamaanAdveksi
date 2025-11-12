import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameter
nx = 200          # jumlah grid
Lx = 10.0         # panjang domain (m)
dx = Lx / nx
v = 0.5           # kecepatan adveksi (m/s)

CFL = 0.8         # Courant number (<=1 agar stabil)
dt = CFL * dx / abs(v)
nt = 200          # jumlah langkah waktu

x = np.linspace(0, Lx, nx)

# Kondisi awal: Gaussian di tengah
u = np.exp(-((x - 3.0)**2) / 0.2)

# Skema upwind 1D
def upwind_1d(u, v, dt, dx):
    un = u.copy()
    unew = np.zeros_like(u)
    for i in range(nx):
        if v > 0:
            unew[i] = un[i] - v * dt/dx * (un[i] - un[i-1])
        else:
            ip = i+1 if i+1 < nx else i
            unew[i] = un[i] - v * dt/dx * (un[ip] - un[i])
    return unew

# Visualisasi
fig, ax = plt.subplots()
line, = ax.plot(x, u, lw=2, color='blue')
ax.set_ylim(-0.1, 1.1)
ax.set_xlim(0, Lx)
ax.set_xlabel('Posisi x (m)')
ax.set_ylabel('Konsentrasi C')
title = ax.set_title('Simulasi Adveksi 1D (Skema Upwind)')

# Fungsi animasi
def animate(n):
    global u
    u = upwind_1d(u, v, dt, dx)
    line.set_ydata(u)
    title.set_text(f"Simulasi Adveksi 1D (t = {n*dt:.2f} s)")
    return line,

# Jalankan animasi
ani = FuncAnimation(fig, animate, frames=nt, interval=50)
plt.show()
