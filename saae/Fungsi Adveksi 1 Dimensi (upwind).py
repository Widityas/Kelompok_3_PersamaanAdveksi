import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

L =2.0                   # panjang domain (m)
nx = 200                 # jumlah grid
dx = L / (nx-1)          # jarak antar grid

v = 0.1                  # kecepatan adveksi (m/s)
dt = 0.01                # langkah waktu (s)
CFL = v * dt / dx        # Courant number

print("CFL =", CFL)
if CFL > 1:
    print("WARNING: CFL > 1 --> solusi akan instabil")

x = np.linspace(0, L, nx)
u = np.exp(-100* (x - 0.3)**2)  # kondisi awal: Gaussian

def adveksi_1d_upwind(u, v, dt, dx):
    un = u.copy()
    unew = np.zeros_like(u)
    for i in range(nx):
        unew[i] = un[i] - v * dt/dx * (un[i] - un[i-1]) # upwind scheme       
    return unew

fig, ax = plt.subplots()
line, = ax.plot(x, u, label = 'Konsentrasi', color='magenta') # inisialisasi plot

ax.set_xlim(0, L)                                           # batas sumbu x
ax.set_ylim(-0.1, 1.1)                                      # batas sumbu y
ax.set_xlabel('Posisi (m)')                                 # label sumbu x  
ax.set_ylabel('Konsentrasi')                                # label sumbu y
ax.set_title('Adveksi 1 Dimensi (Upwind Scheme)')           # judul grafik
ax.legend()                                                 # menampilkan legenda
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes) # menampilkan waktu simulasi 

def update(frame):                                          # fungsi update untuk animasi
    global u                                                # gunakan variabel global u
    u = adveksi_1d_upwind(u, v, dt, dx)                     # hitung adveksi
    line.set_ydata(u)                                       # perbarui data y pada plot
    time_text.set_text(f'Waktu = {frame*dt:.2f} s')         # perbarui teks waktu
    return line, time_text                                  # kembalikan objek yang diperbarui

ani = FuncAnimation(fig, update, frames=200, blit=True, interval=30) # buat animasi

# Simpan sebagai MP4
# ===============================
# ANIMASI & SAVE VIDEO
# ===============================
ani.save('hasil_adveksi_surface.mp4', writer='ffmpeg', fps=20, dpi=150) 
plt.show()