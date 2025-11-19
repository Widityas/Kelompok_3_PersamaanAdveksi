import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =======================================================
# DOMAIN & GRID
# =======================================================
nx, ny = 121, 121
xmin, xmax = 0.0, 10.0
ymin, ymax = 0.0, 10.0

x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
dx = (xmax - xmin) / (nx - 1)
dy = (ymax - ymin) / (ny - 1)

X, Y = np.meshgrid(x, y)

# =======================================================
# KECEPATAN GELOMBANG (2 gelombang saling mendekat)
# =======================================================
vx1, vy1 = 1.0, 0.0      # gelombang pertama → kanan
vx2, vy2 = -1.0, 0.0     # gelombang kedua → kiri

eps = 1e-12
dt = dx / max(abs(vx1), abs(vx2), eps) * 0.4
nt = 200

# =======================================================
# KONDISI AWAL: 2 Gaussian di kiri dan kanan
# =======================================================
u1 = np.exp(-((X - 3.0)**2 + (Y - 5.0)**2) / 1.2)
u2 = np.exp(-((X - 7.0)**2 + (Y - 5.0)**2) / 1.2)

u = u1 + u2

# =======================================================
# SKEMA UPWIND 2D
# =======================================================
def upwind_2d(u, vx, vy):
    if vx > 0:
        dudx = (u - np.roll(u, 1, axis=1)) / dx
    else:
        dudx = (np.roll(u, -1, axis=1) - u) / dx

    if vy > 0:
        dudy = (u - np.roll(u, 1, axis=0)) / dy
    else:
        dudy = (np.roll(u, -1, axis=0) - u) / dy

    return u - dt * (vx * dudx + vy * dudy)

# =======================================================
# SETUP ANIMASI
# =======================================================
fig, ax = plt.subplots(figsize=(7, 6))
contour = ax.contourf(X, Y, u, 40, cmap="turbo")
plt.colorbar(contour)

ax.set_title("Dua Gelombang 2D Bertabrakan (t = 0 s)")
ax.set_xlabel("x")
ax.set_ylabel("y")

# =======================================================
# FUNGSI PEMBARUAN FRAME
# =======================================================
def update(frame):
    global u1, u2, u

    # update masing-masing gelombang
    u1 = upwind_2d(u1, vx1, vy1)
    u2 = upwind_2d(u2, vx2, vy2)

    # superposisi fisik
    u = u1 + u2

    ax.clear()
    contour = ax.contourf(X, Y, u, 40, cmap="turbo")
    ax.set_title(f"Dua Gelombang 2D Bertabrakan (t = {frame*dt:.2f} s)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return contour

# =======================================================
# ANIMASI
# =======================================================
anim = FuncAnimation(fig, update, frames=nt, interval=60, repeat=False)
plt.show()
