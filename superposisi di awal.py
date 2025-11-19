import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==============================
# Parameter grid dan fisika
# ==============================
nx = 800
Lx = 20.0          # lintasan lebih panjang
dx = Lx / nx

v1 = -1.0          # gelombang 1 ke kiri
v2 =  1.0          # gelombang 2 ke kanan

CFL = 0.8
dt = CFL * dx / max(abs(v1), abs(v2))
nt = 500

x = np.linspace(0, Lx, nx)

# ==============================
# Kondisi awal: Superposisi di tengah,
# amplitudo diturunkan supaya tidak terlalu tinggi
# ==============================
center = Lx / 2
amp = 0.5           # turunkan amplitudo agar puncak superposisi tidak terlalu tinggi

u1 = amp * np.exp(-((x - center)**2) / 0.3)
u2 = amp * np.exp(-((x - center)**2) / 0.3)

u = u1 + u2   # superposisi awal

# ==============================
# Skema upwind 1D
# ==============================
def upwind_1d(u, v, dt, dx):
    un = u.copy()
    unew = np.zeros_like(u)
    for i in range(nx):
        if v > 0:
            im = i - 1 if i - 1 >= 0 else i
            unew[i] = un[i] - v*dt/dx*(un[i] - un[im])
        else:
            ip = i + 1 if i + 1 < nx else i
            unew[i] = un[i] - v*dt/dx*(un[ip] - un[i])
    return unew

# ==============================
# Plot setup
# ==============================
fig, ax = plt.subplots(figsize=(10, 4))

line1, = ax.plot(x, u1, lw=2, color='red', label='Gelombang 1')
line2, = ax.plot(x, u2, lw=2, color='blue', label='Gelombang 2')
line3, = ax.plot(x, u,  lw=2, color='black', label='Superposisi')

ax.set_xlim(0, Lx)
ax.set_ylim(-0.1, 1.1 * amp * 2)  # superposisi terlihat utuh tanpa terlalu tinggi
ax.set_xlabel("Posisi x (m)")
ax.set_ylabel("Amplitudo")
ax.legend()
title = ax.set_title("Dua Gelombang Memisah (t = 0 s)")

# ==============================
# Animasi
# ==============================
def animate(n):
    global u1, u2, u

    u1 = upwind_1d(u1, v1, dt, dx)
    u2 = upwind_1d(u2, v2, dt, dx)
    u = u1 + u2

    line1.set_ydata(u1)
    line2.set_ydata(u2)
    line3.set_ydata(u)

    title.set_text(f"Dua Gelombang Memisah (t = {n*dt:.2f} s)")
    return line1, line2, line3

ani = FuncAnimation(fig, animate, frames=nt, interval=30, repeat=False)
plt.show()
