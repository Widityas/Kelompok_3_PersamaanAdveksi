import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ===============================
# PARAMETER DOMAIN
# ===============================
L = 2.0
nx = 400
dx = L / (nx - 1)
x = np.linspace(0, L, nx)

dt = 0.005
v1 = 0.7     # pulse belakang (lebih cepat)
v2 = 0.1     # pulse depan (lebih lambat)

# flag apakah sudah tabrakan
collided = False

# ===============================
# KONDISI AWAL
# ===============================
u1 = np.exp(-200 * (x - 0.4)**2)   # pulse belakang
u2 = np.exp(-200 * (x - 1.2)**2)   # pulse depan

# ===============================
# SCHEME UPWIND
# ===============================
def upwind(u, v, dt, dx):
    un = u.copy()
    unew = np.zeros_like(u)
    for i in range(1, len(u)):
        unew[i] = un[i] - v * dt/dx * (un[i] - un[i-1])
    return unew


# ===============================
# FIGURE
# ===============================
fig, ax = plt.subplots()
line, = ax.plot(x, u1 + u2, color='magenta')
ax.set_ylim(-0.1, 1.2)
ax.set_xlim(0, L)
ax.set_title("Dua Pulse: Menyusul → Tabrak → Menyatu")
time_text = ax.text(0.05, 0.92, "", transform=ax.transAxes)


# ===============================
# ANIMASI
# ===============================
def update(frame):
    global u1, u2, collided

    if not collided:
        # Update masing-masing pulse
        u1 = upwind(u1, v1, dt, dx)
        u2 = upwind(u2, v2, dt, dx)

        # cek tabrakan: ketika superposisi mencapai nilai maks
        if np.max(u1 + u2) > 0.95:
            collided = True
            # Buat pulse baru → campuran keduanya
            u_combined = 0.5*(u1 + u2)
            update.u = u_combined     # simpan dalam fungsi
    else:
        # setelah nabrak → pulse bergerak bersama
        update.u = upwind(update.u, v1, dt, dx)

    # Update visual
    if collided:
        line.set_ydata(update.u)
    else:
        line.set_ydata(u1 + u2)

    time_text.set_text(f"t = {frame*dt:.3f} s")
    return line, time_text


ani = FuncAnimation(fig, update, frames=500, interval=30, blit=True)
plt.show()