import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ================== Parameter ==================
nx, ny = 100, 100       # jumlah grid
Lx, Ly = 10.0, 10.0     # panjang domain
dx, dy = Lx/nx, Ly/ny
u, v = 1.0, 1.0          # kecepatan adveksi (m/s)
CFL = 0.4
dt = CFL * min(dx/abs(u), dy/abs(v))
nt = 150

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# ================== Kondisi Awal (Gaussian) ==================
C = np.exp(-((X - 3.0)**2 + (Y - 3.0)**2) / 0.5)

# ================== Fungsi Adveksi (Upwind 2D) ==================
def upwind_2d(C, u, v, dt, dx, dy):
    Cn = C.copy()
    Cnew = np.zeros_like(C)
    for i in range(nx):
        for j in range(ny):
            im = (i - 1) % nx
            jm = (j - 1) % ny
            if u > 0 and v > 0:
                Cnew[j, i] = (Cn[j, i] 
                              - u * dt/dx * (Cn[j, i] - Cn[j, im])
                              - v * dt/dy * (Cn[j, i] - Cn[jm, i]))
            elif u > 0 and v < 0:
                jp = (j + 1) % ny
                Cnew[j, i] = (Cn[j, i] 
                              - u * dt/dx * (Cn[j, i] - Cn[j, im])
                              - v * dt/dy * (Cn[jp, i] - Cn[j, i]))
            elif u < 0 and v > 0:
                ip = (i + 1) % nx
                Cnew[j, i] = (Cn[j, i] 
                              - u * dt/dx * (Cn[j, ip] - Cn[j, i])
                              - v * dt/dy * (Cn[j, i] - Cn[jm, i]))
            else:
                ip = (i + 1) % nx
                jp = (j + 1) % ny
                Cnew[j, i] = (Cn[j, i] 
                              - u * dt/dx * (Cn[j, ip] - Cn[j, i])
                              - v * dt/dy * (Cn[jp, i] - Cn[j, i]))
    return Cnew

# ================== Setup Plot ==================
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(C, origin='lower', extent=[0, Lx, 0, Ly],
               cmap='viridis', vmin=0, vmax=1.5)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Konsentrasi C')
ax.set_xlabel('Posisi X')
ax.set_ylabel('Posisi Y')
title = ax.set_title('Adveksi 2D dengan Kondisi Batas Terbuka\nWaktu 0.0 s')

# ================== Fungsi Animasi ==================
def animate(n):
    global C
    C = upwind_2d(C, u, v, dt, dx, dy)
    im.set_data(C)
    title.set_text(f'Adveksi 2D dengan Kondisi Batas Terbuka\nWaktu {n*dt:.3f} s')
    return [im]

ani = FuncAnimation(fig, animate, frames=nt, interval=80, blit=True)
plt.show()
