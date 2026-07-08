import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio

dataset_path = '/home/atkesonlab4/Desktop/YiqiProject/100%_Project/ethernet-plugging/logs-collectfailures/75rl-rtc/episode000001'
states = np.load(f'{dataset_path}/states.npz')
forces = states['gripper_force']
eef_forces = states['force'][:, :3]

image_dir = os.path.join(dataset_path, "images")
images, N = [], len(os.listdir(image_dir))
for idx in range(N):
    image_path = os.path.join(image_dir, f"{idx:06d}.png")
    images.append(np.array(Image.open(image_path)))
N = 10 #  min(N, len(forces))
image_H, image_W = images[0].shape[:2]

# ============================================================
# Visualization settings
# ============================================================
window_size = 20

force_min, force_max = np.min(forces), np.max(forces)
eef_force_min, eef_force_max = np.min(eef_forces), np.max(eef_forces)
frames = []

# ============================================================
# Generate frames
# ============================================================
for idx in tqdm(range(N), desc="Generating frames"):

    fig = plt.figure(figsize=(10, 5))

    # --------------------------------------------------------
    # Left: RGB image
    # --------------------------------------------------------
    ax1 = fig.add_subplot(1, 2, 1)

    ax1.imshow(images[idx])
    ax1.set_title(f"Frame {idx}")
    ax1.axis("off")

    # --------------------------------------------------------
    # Right: Force history
    # --------------------------------------------------------
    ax2 = fig.add_subplot(1, 2, 2)

    start_idx = max(0, idx - window_size + 1)

    x = np.arange(start_idx, idx + 1)
    y = forces[start_idx:idx + 1]

    line1 = ax2.plot(x, y, color = 'black', label="Gripper Force")

    ax2.set_xlim(
        max(0, idx - window_size + 1),
        max(window_size, idx + 1),
    )

    # Fixed y-axis across entire episode
    ax2.set_ylim(force_min, force_max)

    ax2.set_title("Gripper Force")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Force")

    ax2.grid(True)

    # Create second y-axis for x-y-z force
    ax2b = ax2.twinx()

    eef_y = eef_forces[start_idx:idx + 1]  # shape: (N, 3)

    line2 = ax2b.plot(x, eef_y[:, 0], label="Fx")
    line3 = ax2b.plot(x, eef_y[:, 1], label="Fy")
    line4 = ax2b.plot(x, eef_y[:, 2], label="Fz")

    ax2b.set_ylim(eef_force_min, eef_force_max)
    ax2b.set_ylabel("EEF Force")

    # Combine legends from both axes
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper left")

    # --------------------------------------------------------
    # Convert matplotlib figure -> numpy image
    # --------------------------------------------------------
    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    frame = np.asarray(canvas.buffer_rgba())[..., :3]

    frames.append(frame)

    plt.close(fig)

# ============================================================
# Save GIF
# ============================================================
gif_path = os.path.join(dataset_path, "force_visualization.gif")

imageio.mimsave(
    gif_path,
    frames,
    fps=20,
    loop=0, # infinite loop
)

print(f"Saved GIF to: {gif_path}")