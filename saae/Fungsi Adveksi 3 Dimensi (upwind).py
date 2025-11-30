import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# PARAMETER DOMAIN
Lx = 2.0; Ly = 2.0
nx = 80 ;ny = 80

dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

Xvel = 1.0     # dibesarkan agar terlihat bergerak
Yvel = 0.5
dt = 0.01      # langkah waktu diperbesar

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)
# KONDISI AWAL
U = np.exp(-50 * ((X - 0.6)**2 + (Y - 0.6)**2))

# ADVESI UPWIND
def adveksi_2d_upwind(U, vx, vy, dt, dx, dy):
    Un = U.copy()
    Unew = np.zeros_like(U)
    nx, ny = U.shape[1], U.shape[0]

    for i in range(nx):
        for j in range(ny):
            im = i - 1 if i > 0 else nx - 1
            jm = j - 1 if j > 0 else ny - 1

            Unew[j, i] = (Un[j, i]
                          - vx * dt/dx * (Un[j, i] - Un[j, im])
                          - vy * dt/dy * (Un[j, i] - Un[jm, i]))
    return Unew

# SETUP PLOT
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, U, cmap='viridis')
ax.set_zlim(0, 1)
ax.set_title("Adveksi 2D dengan Amplitudo" )
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Konsentrasi")

time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

# ANIMASI
def update(frame):
    global U, surf

    # gerakkan fungsi
    U = adveksi_2d_upwind(U, Xvel, Yvel, dt, dx, dy)

    # hapus permukaan lama & plot ulang
    surf.remove()
    surf = ax.plot_surface(X, Y, U, cmap='viridis')

    time_text.set_text(f"Time = {frame * dt:.3f} s")
    return surf, time_text

ani = FuncAnimation(fig, update, frames=100, interval=30, blit=False)
ani.save("Adveksi3D.gif", writer="pillow", fps=20) 
plt.show()