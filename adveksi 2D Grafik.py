import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==============================
# Parameter domain dan grid
# ==============================
nx = 150
ny = 150
Lx = 10.0
Ly = 10.0

dx = Lx / nx
dy = Ly / ny

vx = 1.0    # kecepatan adveksi arah x
vy = 0.5    # kecepatan adveksi arah y

CFL = 0.8
dt = CFL * min(dx / abs(vx), dy / abs(vy))
nt = 200    # jumlah langkah waktu

# Meshgrid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# ==============================
# Kondisi awal (Gaussian 2D)
# ==============================
u = np.exp(-((X - 4.0)**2 + (Y - 5.0)**2) / 0.8)

# ==============================
# Skema upwind 2D
# ==============================
def upwind_2d(u, vx, vy, dt, dx, dy):
    un = u.copy()
    unew = np.zeros_like(u)

    for j in range(ny):
        for i in range(nx):

            # indeks tetangga untuk upwind
            im = i - 1 if i - 1 >= 0 else i
            jm = j - 1 if j - 1 >= 0 else j

            # upwind arah x
            if vx > 0:
                dudx = (un[i, j] - un[im, j]) / dx
            else:
                ip = i + 1 if i + 1 < nx else i
                dudx = (un[ip, j] - un[i, j]) / dx

            # upwind arah y
            if vy > 0:
                dudy = (un[i, j] - un[i, jm]) / dy
            else:
                jp = j + 1 if j + 1 < ny else j
                dudy = (un[i, jp] - un[i, j]) / dy

            # persamaan adveksi 2D
            unew[i, j] = un[i, j] - dt * (vx * dudx + vy * dudy)

    return unew

# ==============================
# Visualisasi
# ==============================
fig, ax = plt.subplots()
img = ax.imshow(u.T, origin="lower", extent=[0, Lx, 0, Ly], cmap="viridis")
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
title = ax.set_title("Adveksi 2D (t = 0 s)")
plt.colorbar(img, ax=ax)

# ==============================
# Fungsi animasi
# ==============================
def animate(n):
    global u
    u = upwind_2d(u, vx, vy, dt, dx, dy)
    img.set_data(u.T)
    title.set_text(f"Adveksi 2D (t = {n*dt:.2f} s)")
    return img,

ani = FuncAnimation(fig, animate, frames=nt, interval=50)
plt.show()
