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
v1 = 0.2               # kecepatan pulse 1 (lebih cepat)
v2 = 0.7                    #  kecepatan pulse 2 (lebih lambat)

# koefisien restitusi:
# 1.0 = elastis (tidak bercampur)
# 0.5 = inelastis sebagian
# 0.0 = menyatu total
e = 0


# ===============================
# KONDISI AWAL
# ===============================
u1 = np.exp(-200 * (x - 1.2)**2)   # pulse depan
u2 = np.exp(-200 * (x - 0.4)**2)   # pulse belakang


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
# SETUP FIGURE
# ===============================
fig, ax = plt.subplots()
line, = ax.plot(x, u1 + u2, color='magenta')
ax.set_ylim(-0.1, 1.2)
ax.set_xlim(0, L)
ax.set_xlabel("Posisi")
ax.set_ylabel("Amplitudo")
ax.set_title(f"Adveksi 1D - Dua Pulsa Saling Menyusul (e = {e})")

time_text = ax.text(0.02, 0.92, "", transform=ax.transAxes)


# ===============================
# UPDATE FRAME ANIMASI
# ===============================
def update(frame):
    global u1, u2

    u1 = upwind(u1, v1, dt, dx)
    u2 = upwind(u2, v2, dt, dx)

    # model tabrakan / mixing
    u_mix = e * u1 + e * u2 + (1 - e) * 0.5 * (u1 + u2)

    line.set_ydata(u_mix)
    time_text.set_text(f"t = {frame*dt:.2f} s")
    return line, time_text


# ===============================
# ANIMASI
# ===============================
ani = FuncAnimation(fig, update, frames=400, interval=30, blit=True)
plt.show()
