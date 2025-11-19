import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==============================
# Parameter grid dan fisika
# ==============================
nx = 400
Lx = 10.0
dx = Lx / nx

v1 = 1.0      # kecepatan gelombang 1 (bergerak ke kanan)
v2 = -1.0     # kecepatan gelombang 2 (bergerak ke kiri)

CFL = 0.8
dt = CFL * dx / max(abs(v1), abs(v2))
nt = 400      # jumlah langkah waktu

x = np.linspace(0, Lx, nx)

# ==============================
# Kondisi awal: dua Gaussian
# ==============================
u1 = np.exp(-((x - 3.0)**2) / 0.2)   # gelombang kiri → kanan
u2 = np.exp(-((x - 7.0)**2) / 0.2)   # gelombang kanan → kiri

# superposisi awal
u = u1 + u2

# ==============================
# Skema upwind
# ==============================
def upwind_1d(u, v, dt, dx):
    un = u.copy()
    unew = np.zeros_like(u)
    for i in range(nx):
        if v > 0:
            im = i - 1 if i - 1 >= 0 else i
            unew[i] = un[i] - v * dt/dx * (un[i] - un[im])
        else:
            ip = i + 1 if i + 1 < nx else i
            unew[i] = un[i] - v * dt/dx * (un[ip] - un[i])
    return unew

# ==============================
# Plot setup
# ==============================
fig, ax = plt.subplots()
line1, = ax.plot(x, u1, lw=2, color='red', label='Gelombang 1')
line2, = ax.plot(x, u2, lw=2, color='blue', label='Gelombang 2')
line3, = ax.plot(x, u,  lw=2, color='black', label='Superposisi')

ax.set_ylim(-0.1, 1.2)
ax.set_xlim(0, Lx)
ax.set_xlabel("Posisi x (m)")
ax.set_ylabel("Amplitudo")
ax.legend()
title = ax.set_title("Dua Gelombang Bertabrakan (t = 0 s)")

# ==============================
# Fungsi animasi
# ==============================
def animate(n):
    global u1, u2, u

    # Update masing-masing gelombang
    u1 = upwind_1d(u1, v1, dt, dx)
    u2 = upwind_1d(u2, v2, dt, dx)

    # Superposisi fisik
    u = u1 + u2

    # Update plot
    line1.set_ydata(u1)
    line2.set_ydata(u2)
    line3.set_ydata(u)

    title.set_text(f"Dua Gelombang Bertabrakan (t = {n*dt:.2f} s)")
    return line1, line2, line3

# ==============================
# Jalankan animasi (tanpa looping)
# ==============================
ani = FuncAnimation(fig, animate, frames=nt, interval=40, repeat=False)
plt.show()
