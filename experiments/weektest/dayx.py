import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ========== 定義矩陣 ==========
theta = np.deg2rad(45)
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
S = np.array([[1, 0],
              [0, 2]])

# R×S 與 S×R 兩種順序
RS = R @ S
SR = S @ R

# ========== 準備方格 ==========
x = np.linspace(-1, 1, 10)
y = np.linspace(-1, 1, 10)
X, Y = np.meshgrid(x, y)
grid = np.stack([X.flatten(), Y.flatten()], axis=0)

# ========== 定義動畫更新函數 ==========
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
titles = ["R × S (先拉伸再旋轉)", "S × R (先旋轉再拉伸)"]
mats = [RS, SR]
scatters = []

for ax, title in zip(axes, titles):
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_title(title, fontsize=12)
    scat = ax.scatter([], [], s=10, color="royalblue")
    scatters.append(scat)

def update(frame):
    t = frame / 20
    for scat, M in zip(scatters, mats):
        interp = np.eye(2) * (1 - t) + M * t
        grid_t = interp @ grid
        scat.set_offsets(grid_t.T)
    return scatters

ani = FuncAnimation(fig, update, frames=21, interval=100, blit=True)
plt.show()
