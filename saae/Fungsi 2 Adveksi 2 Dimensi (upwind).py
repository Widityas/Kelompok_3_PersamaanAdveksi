import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# 1. PARAMETER DOMAIN
Lx = 4.0                    # Panjang X disamakan dengan kode 1D (agar lega)
Ly = 2.0                    # Lebar Y
nx = 200                    # Resolusi grid X
ny = 100                    # Resolusi grid Y (dikurangi dikit biar render cepat)

dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

# Kecepatan (Sama seperti 1D, tapi dalam vektor)
# Pulse 1 (Lambat, di depan)
v1_x = 0.2
v1_y = 0.0                 
# Pulse 2 (Cepat, di belakang)
v2_x = 0.7
v2_y = 0.0

dt = 0.005                  # Langkah waktu

# Cek stabilitas (CFL)
v_max = max(v1_x, v2_x)
CFL = v_max * dt / dx
print(f"CFL number = {CFL:.4f}")
if CFL > 1:
    print("WARNING: CFL > 1 (Tidak Stabil!) Kurangi dt atau perbesar dx.")

# Grid 2D
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# KONDISI AWAL (2 GAUSSIAN)
U1 = np.exp(-100 * ((X - 1.2)**2 + (Y - 1.0)**2))
U2 = np.exp(-100 * ((X - 0.4)**2 + (Y - 1.0)**2))

# 3. FUNGSI ADVEKSI (VECTORIZED)
def adveksi_cepat(U, vx, vy, dt, dx, dy):
    Un = U.copy()
    
    U_kiri  = np.roll(Un, 1, axis=1) # Geser ke kanan (ambil nilai kiri)
    U_bawah = np.roll(Un, 1, axis=0) # Geser ke bawah (ambil nilai atas/bawah)
    
    U_kiri[:, 0] = 0
    U_bawah[0, :] = 0
    
    cx = vx * dt / dx
    cy = vy * dt / dy
    
    Unew = Un - cx * (Un - U_kiri) - cy * (Un - U_bawah)
    return Unew
# SETUP PLOT (IMSHOW)
fig, ax = plt.subplots(figsize=(10, 5))

# Hitung kondisi awal gabungan
U_total = U1 + U2

im = ax.imshow(U_total, origin='lower',
               extent=[0, Lx, 0, Ly],
               cmap='viridis',       # Warna api (bagus untuk intensitas)
               vmin=0, vmax=1.5,     # vmax 1.5 karena saat tumpuk tingginya > 1
               interpolation='bicubic') # Biar terlihat halus

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Amplitudo')

ax.set_title("Adveksi 2D - Dua Pulsa Saling Menyusul")
ax.set_xlabel("Posisi X")
ax.set_ylabel("Posisi Y")

time_text = ax.text(0.02, 0.92, "", transform=ax.transAxes, color='white', fontweight='bold')
# 5. UPDATE ANIMASI
def update(frame):
    global U1, U2
    
    U1 = adveksi_cepat(U1, v1_x, v1_y, dt, dx, dy)
    U2 = adveksi_cepat(U2, v2_x, v2_y, dt, dx, dy)
    U_mix = U1 + U2
    im.set_data(U_mix)
    time_text.set_text(f"Time = {frame * dt:.2f} s")
    return im, time_text

ani = FuncAnimation(fig, update, frames=1200, interval=1, blit=True)
ani.save('2 Adveksi 2 Dimensi.mp4', writer='ffmpeg', fps=60, dpi=150)
plt.show()