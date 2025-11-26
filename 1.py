import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameter
v = 1.0
x = np.linspace(-10, 10, 400)

# Fungsi solusi adveksi
def u(x, t, v):
    return np.exp(-(x - v*t)**2)

# Setup figure
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(-10, 10)
ax.set_ylim(0, 1.1)
ax.set_xlabel("x (ruang)")
ax.set_ylabel("u(x,t)")
ax.set_title("Animasi Adveksi 1D: Gaussian Bergeser")

# Inisialisasi frame kosong
def init():
    line.set_data([], [])
    return line,

# Update tiap frame
def animate(t):
    y = u(x, t, v)
    line.set_data(x, y)
    return line,

# Buat animasi
ani = animation.FuncAnimation(fig, animate, frames=np.linspace(0, 5, 100),
                              init_func=init, blit=True, interval=100)

plt.show()
