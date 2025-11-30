import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# ===============================
# PARAMETER & DOMAIN
# ===============================
Lx = 3.0
Ly = 2.5
nx = 60  # Dikurangi agar animasi lancar (surface plot berat dirender)
ny = 60

dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

dt = 0.01

# Kecepatan (Hanya bergerak di sumbu X agar saling menyusul seperti kasus 1D)
# Pulse 1 (Lambat, di depan) - tapi di kode asli v1=0.2 (posisi 1.2)
# Pulse 2 (Cepat, di belakang) - di kode asli v2=0.7 (posisi 0.4)
v1_x = 0.2
v1_y = 0.0

v2_x = 0.7
v2_y = 0.0

# Koefisien restitusi (e)
# 1.0 = Elastis (tumpuk biasa)
# 0.0 = Menyatu total (amplitudo rata-rata)
e = 0.0

# ===============================
# KONDISI AWAL (2D GAUSSIAN)
# ===============================
# Pulse 1: Posisi X=1.2, Y=Tengah
U1 = np.exp(-100 * ((X - 1.2)**2 + (Y - 1.0)**2))

# Pulse 2: Posisi X=0.4, Y=Tengah   
U2 = np.exp(-100 * ((X - 0.4)**2 + (Y - 1.0)**2))

# ===============================
# FUNGSI ADVEKSI 2D (UPWIND)
# ===============================
def adveksi_2d_upwind(U, vx, vy, dt, dx, dy):
    Un = U.copy()
    Unew = np.zeros_like(U)
    ny, nx = U.shape

    # Menggunakan slicing numpy agar jauh lebih cepat daripada loop for
    # (Loop manual sangat lambat untuk animasi surface)
    
    # Geser untuk i-1 (kiri)
    U_left = np.roll(Un, 1, axis=1) 
    U_left[:, 0] = Un[:, 0] # Boundary condition sederhana
    
    # Geser untuk j-1 (bawah)
    U_down = np.roll(Un, 1, axis=0)
    U_down[0, :] = Un[0, :] # Boundary condition sederhana

    # Rumus Upwind: u_new = u - cx*(u - u_left) - cy*(u - u_down)
    cx = vx * dt / dx
    cy = vy * dt / dy
    
    Unew = Un - cx * (Un - U_left) - cy * (Un - U_down)
    
    return Unew

# ===============================
# SETUP PLOT 3D
# ===============================
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Hitung kondisi awal campuran
U_mix = e * U1 + e * U2 + (1 - e) * 0.5 * (U1 + U2)

surf = ax.plot_surface(X, Y, U_mix, cmap='viridis', rstride=2, cstride=2)

ax.set_zlim(0, 1.0)
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
ax.set_xlabel("Posisi X")
ax.set_ylabel("Posisi Y")
ax.set_zlabel("Amplitudo")
ax.set_title(f"Adveksi 2D - Tabrakan Pulsa (e={e})")

time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

# ===============================
# UPDATE ANIMASI
# ===============================
def update(frame):
    global U1, U2, surf

    # Update posisi masing-masing pulse
    U1 = adveksi_2d_upwind(U1, v1_x, v1_y, dt, dx, dy)
    U2 = adveksi_2d_upwind(U2, v2_x, v2_y, dt, dx, dy)

    # Model Tabrakan / Mixing (Logika sama dengan kode 1D)
    U_mix = e * U1 + e * U2 + (1 - e) * 0.5 * (U1 + U2)

    # Hapus surface lama dan plot yang baru
    surf.remove()
    surf = ax.plot_surface(X, Y, U_mix, cmap='viridis', rstride=1, cstride=1)

    time_text.set_text(f"t = {frame * dt:.2f} s")
    return surf, time_text

# Interval diperbesar sedikit karena render 3D lebih berat
ani = FuncAnimation(fig, update, frames=300, interval=50, blit=False)
plt.show()