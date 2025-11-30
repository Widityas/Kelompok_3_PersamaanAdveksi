import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# PARAMETER & DOMAIN
Lx = 4.0; Ly = 1.5
nx = 60 ; ny = 60

dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)
dt = 0.01

v1_x = 0.2; v1_y = 0.0
v2_x = 0.7; v2_y = 0.0

# KONDISI AWAL (2D GAUSSIAN)
U1 = np.exp(-100 * ((X - 1.2)**2 + (Y - 1.0)**2))
U2 = np.exp(-100 * ((X - 0.4)**2 + (Y - 1.0)**2))

# FUNGSI ADVEKSI 2D (UPWIND)
def adveksi_2d_upwind(U, vx, vy, dt, dx, dy):
    Un = U.copy()
    Unew = np.zeros_like(U)
    ny, nx = U.shape
    
    U_left = np.roll(Un, 1, axis=1) 
    U_left[:, 0] = Un[:, 0] # Boundary condition sederhana
    
    U_down = np.roll(Un, 1, axis=0)
    U_down[0, :] = Un[0, :] # Boundary condition sederhana

    cx = vx * dt / dx
    cy = vy * dt / dy
    
    Unew = Un - cx * (Un - U_left) - cy * (Un - U_down)
    return Unew

# SETUP PLOT 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Hitung kondisi awal campuran
U_mix = (U1 + U2)

surf = ax.plot_surface(X, Y, U_mix, cmap='viridis', rstride=2, cstride=2)

ax.set_zlim(0, 1.0)
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
ax.set_xlabel("Posisi X")
ax.set_ylabel("Posisi Y")
ax.set_zlabel("Amplitudo")
ax.set_title(f"Adveksi 2D dengan Amplitudo")

time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

# UPDATE ANIMASI
def update(frame):
    global U1, U2, surf

    # Update posisi masing-masing pulse
    U1 = adveksi_2d_upwind(U1, v1_x, v1_y, dt, dx, dy)
    U2 = adveksi_2d_upwind(U2, v2_x, v2_y, dt, dx, dy)
    U_mix = (U1 + U2)

    surf.remove()
    surf = ax.plot_surface(X, Y, U_mix, cmap='viridis', rstride=1, cstride=1)

    time_text.set_text(f"t = {frame * dt:.2f} s")
    return surf, time_text

ani = FuncAnimation(fig, update, frames=1200, interval=50, blit=False)
ani.save("2 Adveksi 3 Dimensi.mp4", writer='ffmpeg', fps=20)
plt.show()