import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ===============================
# PARAMETER & DOMAIN
# ===============================
L = 4.0
nx = 400
dx = L / (nx - 1)
x = np.linspace(0, L, nx)

dt = 0.005
v1 = 0.2   # pulse cepat (belakang)
v2 = 0.7   # pulse lambat (depan)

# ===============================
# KONDISI AWAL
# ===============================
u1 = np.exp(-200 * (x - 1.2)**2)
u2 = np.exp(-200 * (x - 0.4)**2)

merged = False
u_merge = None
merge_start_frame = None
merge_duration = 40   # semakin besar → transisi makin smooth

# ===============================
# FUNGSI ADVESI UPWIND
# ===============================
def upwind(u, v, dt, dx):
    un = u.copy()
    unew = np.zeros_like(u)
    for i in range(1, len(u)):
        unew[i] = un[i] - v * dt/dx * (un[i] - un[i-1])
    return unew

# ===============================
# SETUP PLOT
# ===============================
fig, ax = plt.subplots()
line, = ax.plot(x, u1 + u2, color='magenta', lw=2)
ax.set_ylim(-0.1, 2.5)
ax.set_xlim(0, L)
ax.set_xlabel("Posisi")
ax.set_ylabel("Amplitudo")
ax.set_title("Adveksi 1D - Pulsa Menyatu Permanen (Smooth & Stable)")

time_text = ax.text(0.02, 0.92, "", transform=ax.transAxes)

# ===============================
# UPDATE ANIMASI
# ===============================
def update(frame):
    global u1, u2, merged, u_merge, merge_start_frame

    if not merged:
        u1 = upwind(u1, v1, dt, dx)
        u2 = upwind(u2, v2, dt, dx)

        # cek overlap (dua pulse mulai nabrak)
        overlap = np.any((u1 > 0.05) & (u2 > 0.05))

        # kalau baru mulai overlap → mulai merging
        if overlap and merge_start_frame is None:
            merge_start_frame = frame

            # posisi tengah dua peak
            x_center = 0.5 * (x[np.argmax(u1)] + x[np.argmax(u2)])

            # amplitude gabungan (lebih tinggi)
            amp = np.max(u1) + np.max(u2)

            # pulse baru
            u_merge = amp * np.exp(-200 * (x - x_center)**2)

    # fase transisi menuju pulse baru
    if merge_start_frame is not None and frame < merge_start_frame + merge_duration:
        alpha = (frame - merge_start_frame) / merge_duration
        u_mix = (1 - alpha) * (u1 + u2) + alpha * u_merge

    elif merge_start_frame is not None and not merged:
        # setelah transisi selesai → gunakan pulse baru
        merged = True
        u_mix = u_merge
    else:
        u_mix = u1 + u2

    # kalau sudah merged → pulse baru ikut bergerak
    if merged:
        u_merge[:] = upwind(u_merge, (v1 + v2) / 2, dt, dx)
        u_mix = u_merge

    line.set_ydata(u_mix)
    time_text.set_text(f"t = {frame*dt:.2f} s")
    return line, time_text

# ===============================
# ANIMASI
# ===============================
ani = FuncAnimation(fig, update, frames=500, interval=20, blit=False)
plt.show()
