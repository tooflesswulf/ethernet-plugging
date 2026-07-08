import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import torch, imageio
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

pretrained_model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(pretrained_model_name, device_map="auto")
model = AutoModel.from_pretrained(
    pretrained_model_name, 
    device_map="auto", 
)

def get_feature(image):
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs).last_hidden_state.cpu()
    # B x (1 + 4 + 196) x 768
    return outputs[:, 0]

def get_similarity(a, b):
    a_norm = F.normalize(a, p=2, dim=1)  # [N, d]
    b_norm = F.normalize(b, p=2, dim=1)  # [M, d]

    return a_norm @ b_norm.T  

def get_seg_id(gripper_widths, eps=1):
    # Input: N
    # Output: N, monotonically increased integer
    seg_id = 0
    seg_ids = []
    keyframe_idx = []
    features = []
    state = 0 # -1 for small, 0 for unchanged, 1 large
    for i, w in enumerate(gripper_widths):
        if i == 0:
            seg_ids.append(seg_id)
        else:
            # update state 
            prev_w = gripper_widths[i-1]
            if prev_w - eps <= w <= prev_w + eps:
                pass 
            elif w < prev_w - eps:
                # get small
                # assert state <= 0, f"Error at {i}, can't go from state {state} to -1."
                if state >= 0:
                    state = -1
                    seg_id += 1 
                    keyframe_idx.append(i)
                # if state already -1, do nothing
            else:
                # get large 
                # assert state >= 0, f"Error at {i}, can't go from state {state} to 1."
                if state <= 0:
                    state = 1
                    seg_id += 1
                    keyframe_idx.append(i)
                # if state already 1, do nothing
            seg_ids.append(seg_id)
    keyframe_idx.append(i)

    return seg_ids, keyframe_idx

def get_goals(dataset_dir='/home/atkesonlab4/Desktop/YiqiProject/100%_Project/dataset/ethernet_plugin_unplug', N=50):
    # Everytime gripper state changes, get the image
    # return: N x segments x feature_dim
    goals = []
    for n in range(1, N+1):
        episode_path = os.path.join(dataset_dir, f'episode{n:06d}')
        try:
            _, keyframes = get_seg_id( np.load(os.path.join(episode_path, 'states.npz'))['gripper_width'])
        except:
            continue
        ep_keyframes= []
        for id in keyframes:
            image_path = os.path.join(episode_path, 'images', f"{id:06d}.png")
            ep_keyframes.append(
                Image.open(image_path)
            )
        ep_features = get_feature(ep_keyframes)
        goals.append(ep_features.unsqueeze(0))
    goals = torch.cat(goals) # N x segs x d
    return goals
    
s = 3
t = 6
for _ in range(s, s+t):
    dataset_path = f'/home/atkesonlab4/Desktop/YiqiProject/100%_Project/ethernet-plugging/logs-collectfailures/75rl-rtc/episode{_+1:06d}'
    states = np.load(f'{dataset_path}/states.npz')
    widths = states['gripper_width']
    # assert False, f"{widths[::10]}"
    seg_ids, keyframes = get_seg_id(widths)
    goals = get_goals()
    sim2goals = [ ]
    forces = states['gripper_force']
    eef_forces = states['force'][:, :3]

    image_dir = os.path.join(dataset_path, "images")
    images, N = [], len(os.listdir(image_dir))
    for idx in tqdm(range(N)):
        image_path = os.path.join(image_dir, f"{idx:06d}.png")
        img = Image.open(image_path)
        images.append(np.array(img))
        feat = get_feature(img)
        sub_goals = goals[:, seg_ids[idx]]
        sim2goals.append( get_similarity(feat, sub_goals).numpy().max() )
    N =  min([N, len(forces), max(keyframes)])
    sim2goals = np.array(sim2goals)
    image_H, image_W = images[0].shape[:2]

    # ============================================================
    # Visualization settings
    # ============================================================
    window_size = 20

    force_min, force_max = np.min(forces), np.max(forces)
    sim_min, sim_max = np.min(sim2goals), np.max(sim2goals)
    eef_force_min, eef_force_max = np.min(eef_forces), np.max(eef_forces)
    frames = []

    # ============================================================
    # Generate frames
    # ============================================================
    for idx in tqdm(range(N), desc="Generating frames"):

        fig = plt.figure(figsize=(15, 5))

        # --------------------------------------------------------
        # Left: RGB image
        # --------------------------------------------------------
        ax1 = fig.add_subplot(1, 3, 1)

        ax1.imshow(images[idx])
        ax1.set_title(f"Frame {idx}")
        ax1.axis("off")

        # --------------------------------------------------------
        # Middle: Force history
        # --------------------------------------------------------
        ax2 = fig.add_subplot(1, 3, 2)

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

        ax2.set_title("Force")
        ax2.set_xlabel("Timestep")
        ax2.set_ylabel("Gripper orce")

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
        # Right: distance to goal
        # --------------------------------------------------------
        ax3 = fig.add_subplot(1, 3, 3)

        start_idx = max(0, idx - window_size + 1)

        x = np.arange(start_idx, idx + 1)
        y = sim2goals[start_idx:idx + 1]

        # Hide left y-axis
        ax3.yaxis.set_visible(False)
        ax3.spines["left"].set_visible(False)

        # Create right y-axis
        ax3b = ax3.twinx()

        ax3b.plot(x, y, color='green', label=f"Subgoal-{seg_ids[idx]}")

        ax3b.set_xlim(
            max(0, idx - window_size + 1),
            max(window_size, idx + 1),
        )

        # Fixed y-axis across entire episode
        ax3b.set_ylim(sim_min, sim_max)

        ax3b.set_title("Distance to sub goals, DINO")
        ax3b.set_xlabel("Timestep")
        ax3b.set_ylabel("Cosine similarity")
        ax3b.grid(True)
        ax3b.legend()

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