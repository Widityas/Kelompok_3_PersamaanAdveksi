import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Domain
nx, ny = 101, 101
xmin, xmax = 0.0, 10.0
ymin, ymax = 0.0, 10.0

x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
dx = (xmax - xmin) / (nx - 1)
dy = (ymax - ymin) / (ny - 1)

X, Y = np.meshgrid(x, y)

# Kecepatan adveksi
vx = 1.0
vy = 0.5

# Kondisi awal Gaussian
u = np.exp(-((X - 3.0)**2 + (Y - 5.0)**2) / 1.5)

# CFL (diberi eps biar tidak membagi nol)
eps = 1e-8
dt = min(dx / max(abs(vx), eps),
         dy / max(abs(vy), eps)) * 0.5
nt = 150

# ---------------------------------------
# Skema upwind (BEBAS ERROR)
# ---------------------------------------
def upwind(u, vx, vy):
    u_new = u.copy()

    # turunan arah X
    if vx > 0:
        dudx = (u - np.roll(u, 1, axis=1)) / dx
    else:
        dudx = (np.roll(u, -1, axis=1) - u) / dx

    # turunan arah Y
    if vy > 0:
        dudy = (u - np.roll(u, 1, axis=0)) / dy
    else:
        dudy = (np.roll(u, -1, axis=0) - u) / dy

    # update solusi
    u_new = u - dt * (vx * dudx + vy * dudy)
    return u_new


# ---------------------------------------
# Animasi
# ---------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
contour = ax.contourf(X, Y, u, 40, cmap="turbo")
plt.colorbar(contour)

def update(frame):
    global u
    u = upwind(u, vx, vy)
    ax.clear()
    contour = ax.contourf(X, Y, u, 40, cmap="turbo")
    ax.set_title(f"Adveksi 2D – Frame {frame}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return contour

anim = FuncAnimation(fig, update, frames=nt, interval=60)
plt.show()
