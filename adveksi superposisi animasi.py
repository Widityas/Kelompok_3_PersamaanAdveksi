import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =====================================================
# PARAMETER GRID
# =====================================================
Lx, Ly = 30.0, 10.0      # domain lebih panjang
nx, ny = 300, 120
dx, dy = Lx/nx, Ly/ny

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# =====================================================
# KECEPATAN GELOMBANG (berlawanan arah)
# =====================================================
vx1, vy1 = 1.0, 0.0     # gelombang 1 ke kanan
vx2, vy2 = -1.0, 0.0    # gelombang 2 ke kiri

eps = 1e-6
dt = 0.3 * min(dx / (abs(vx1)+eps),
               dx / (abs(vx2)+eps))

# =====================================================
# INISIALISASI GELOMBANG (puncak dibuat lebih rendah)
# =====================================================
u1 = 0.3 * np.exp(-((X - 8.0)**2 + (Y - 5.0)**2) / 2.0)
u2 = 0.3 * np.exp(-((X - 22.0)**2 + (Y - 5.0)**2) / 2.0)

u = u1 + u2

# =====================================================
# SETUP PLOT
# =====================================================
fig, ax = plt.subplots(figsize=(11, 4))
im = ax.imshow(u, origin='lower', cmap='viridis',
               extent=[0, Lx, 0, Ly], vmin=0, vmax=0.6)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Superposisi Dua Gelombang – Adveksi 2D")


# =====================================================
# FUNGSI UPDATE ADVEKSI 2D
# =====================================================
def advect(u, vx, vy):
    dudx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2*dx)
    dudy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2*dy)
    return u - dt * (vx * dudx + vy * dudy)


# =====================================================
# UPDATE ANIMASI
# =====================================================
def update(frame):
    global u1, u2, u

    u1 = advect(u1, vx1, vy1)
    u2 = advect(u2, vx2, vy2)

    u = u1 + u2  # superposisi

    im.set_data(u)
    return [im]


ani = FuncAnimation(fig, update, frames=800, interval=30)
plt.show()
