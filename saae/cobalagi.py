import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ================== Parameter Domain ==================
nx, ny = 150, 150
Lx, Ly = 10.0, 10.0
dx, dy = Lx/nx, Ly/ny

# kecepatan masing-masing pulse — SEARAH
u1, v1 = 0.5, 0.0   # pulse depan (lebih lambat)
u2, v2 = 1.2, 0.0   # pulse belakang (lebih cepat)

CFL = 0.4
dt = CFL * min(dx/abs(u1), dx/abs(u2))
nt = 200

# koefisien restitusi (tabrakan)
e = 0.0   # 0 = menyatu total, 1 = elastis


# ================== Grid ==================
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# ================== Dua Pulsa Gaussian ==================
# satu di depan, satu di belakang
C1 = np.exp(-((X - 3.0)**2 + (Y - 5.0)**2)/0.6)
C2 = np.exp(-((X - 1.5)**2 + (Y - 5.0)**2)/0.6)


# ================== Skema Upwind 2D ==================
def upwind_2d(C, u, v, dt, dx, dy):
    Cn = C.copy()
    Cnew = np.zeros_like(C)
    
    for i in range(nx):
        for j in range(ny):

            im = (i - 1) % nx
            jm = (j - 1) % ny
            ip = (i + 1) % nx
            jp = (j + 1) % ny

            dudx = (Cn[j, i] - Cn[j, im]) if u > 0 else (Cn[j, ip] - Cn[j, i])
            dvdy = (Cn[j, i] - Cn[jm, i]) if v > 0 else (Cn[jp, i] - Cn[j, i])

            Cnew[j, i] = Cn[j, i] - u*dt/dx*dudx - v*dt/dy*dvdy

    return Cnew


# ================== Setup Plot ==================
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(C1 + C2, origin='lower', extent=[0, Lx, 0, Ly],
               cmap='viridis', vmin=0, vmax=1.5)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Konsentrasi')

ax.set_xlabel('X')
ax.set_ylabel('Y')
title = ax.set_title(f"Adveksi 2D - Dua Pulsa Searah Saling Menyusul (e = {e})\nWaktu 0.00 s")

# ================== Animasi ==================
def animate(n):
    global C1, C2

    # kedua pulsa bergerak ke arah yang sama → upwind searah
    C1 = upwind_2d(C1, u1, v1, dt, dx, dy)
    C2 = upwind_2d(C2, u2, v2, dt, dx, dy)

    # model tabrakan (mix)
    Cmix = e*C1 + e*C2 + (1 - e)*0.5*(C1 + C2)

    im.set_data(Cmix)
    title.set_text(f"Adveksi 2D - Dua Pulsa Searah Saling Menyusul (e = {e})\nWaktu {n*dt:.2f} s")

    return [im]

ani = FuncAnimation(fig, animate, frames=nt, interval=50, blit=True)
plt.show()
