import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

Lx = 2.0                   # panjang domain arah x
Ly = 2.0                   # panjang domain arah y
nx = 200                   # jumlah grid x
ny = 200                   # jumlah grid y

dx = Lx / (nx - 1)         # jarak grid arah x
dy = Ly / (ny - 1)         # jarak grid arah y

Xvel = 0.9                 # kecepatan adveksi arah x
Yvel = 0.9                 # kecepatan adveksi arah y
dt = 0.005                 # langkah waktu

CFL = max(Xvel * dt / dx, Yvel * dt / dy)   # Courant-Friedrichs-Lewy number
print("CFL =", CFL)     
if CFL > 1:
    print("WARNING: CFL > 1 --> solusi akan instabil")

x = np.linspace(0, Lx, nx) 
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y) 

# KONDISI AWAL (Gaussian 2D)
U= np.exp(-100 * ((X - 0.5)**2 + (Y - 0.5)**2))

# FUNGSI ADVEKSI UPWIND 2D
def adveksi_2d_upwind(U,Xvel, Yvel, dt, dx, dy):
    Un= U.copy()
    Unew=np.zeros_like(U)

    for i in range(nx):
        for j in range(ny):
            # indeks tetangga kiri
            im = i - 1 if i > 0 else nx - 1
            jm = j - 1 if j > 0 else ny - 1

            Unew[j, i] = (Un[j, i]- Xvel * dt/dx * (Un[j, i] - Un[j, im]) - Yvel * dt/dy * (Un[j, i] - Un[jm, i]) )
    return Unew

# SETUP PLOT
fig, ax = plt.subplots(figsize=(8,6))
im = ax.imshow(U, origin='lower',
               extent=[0, Lx, 0, Ly],
               cmap ='viridis',
               vmin=0, vmax=1,
               interpolation='gaussian') 
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Konsentrasi')

ax.set_title("Adveksi 2D (Upwind Scheme)")
ax.set_xlabel("x")
ax.set_ylabel("y")
time_text = ax.text(0.02, 0.92, "", transform=ax.transAxes, color='white')

# ANIMASI
def update(frame):
    global U
    U = adveksi_2d_upwind(U, Xvel, Yvel, dt, dx, dy)
    im.set_data(U)
    time_text.set_text(f"Waktu = {frame * dt:.3f} s")
    return im, time_text

ani = FuncAnimation(fig, update, frames=200, blit=True, interval=30)
ani.save('hasil_adveksi_surface.mp4', writer='ffmpeg', fps=20, dpi=150) 
plt.show()