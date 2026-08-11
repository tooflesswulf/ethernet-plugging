from PIL import Image
from tqdm import tqdm
import torch, torch.nn as nn, os, cv2, numpy as np
from agent.model.networks import MultiEncoder

import matplotlib.pyplot as plt
import imageio.v2 as imageio

device = 'cuda'
obs_fields=['pose', 'gripper_width', 'force']
IMAGE_SIZE = 128
horizon_steps = 8
img_cond_steps = 1; force_cond_steps = 1; cond_steps = 1

def save_video_with_running_score(
    images,
    scores,
    output_path,
    window_size=50,
    fps=20,
    ylim=None,
):
    """
    Create an MP4 with:
        Left : input images
        Right: running score plot

    Args:
        images: (N,H,W,3) or (N,H,W)
        scores: (N,)
        output_path: output mp4 filename
        window_size: number of recent points to display
        fps: output video fps
        ylim: (ymin, ymax). If None, computed from all scores.
    """
    images = np.asarray(images)
    scores = np.asarray(scores)

    assert len(images) == len(scores)
    N = len(images)

    if ylim is None:
        ymin = 0 # np.min(scores)
        ymax = 1 # np.max(scores)
        pad = 0.05 * (ymax - ymin + 1e-8)
        ylim = (ymin - pad, ymax + pad)

    # Equal-sized panels
    fig, (ax_img, ax_plot) = plt.subplots(
        1,
        2,
        figsize=(8, 4),
        dpi=100,
        gridspec_kw={"width_ratios": [1, 1]},
    )

    writer = None

    for i in range(N):

        ########################
        # Left image
        ########################
        ax_img.clear()

        if images[i].ndim == 2:
            ax_img.imshow(images[i], cmap="gray")
        else:
            ax_img.imshow(images[i])

        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_title(f"Frame {i}")

        ########################
        # Right plot
        ########################
        ax_plot.clear()

        start = max(0, i - window_size + 1)

        x = np.arange(start, i + 1)
        y = scores[start : i + 1]

        ax_plot.plot(
            x,
            y,
            linewidth=2,
        )

        ax_plot.scatter(
            x[-1],
            y[-1],
            color="red",
            s=40,
            zorder=3,
        )

        ax_plot.set_xlim(start, start + window_size - 1)
        ax_plot.set_ylim(*ylim)
        ax_plot.set_xlabel("Frame")
        ax_plot.set_ylabel("Score")
        ax_plot.grid(True)

        plt.tight_layout()

        ########################
        # Convert figure to image
        ########################
        fig.canvas.draw()

        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]

        # Initialize writer once we know frame size
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h),
            )

            if not writer.isOpened():
                raise RuntimeError(f"Failed to open VideoWriter for {output_path}")

        # Matplotlib gives RGB; OpenCV expects BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        writer.write(frame)

    writer.release()
    plt.close(fig)

def get_model(load_path='/home/atkesonlab4/Desktop/YiqiProject/100%_Project/ckpts/failure_detect/best_model.pt'):
    model = MultiEncoder(modality = ['image', 'pose', 'force']).to(device).eval()
    static_dict = torch.load(load_path)
    model.load_state_dict(static_dict)
    return model

def get_image( image_path):
        return np.array( Image.open(image_path).resize((IMAGE_SIZE, IMAGE_SIZE)) ) 

def get_data(ep_dir, j, ep_states):
    image_dir = os.path.join(ep_dir, 'images')
    start = max(0, j-horizon_steps); end = j
    image_indices = np.linspace(start, end, img_cond_steps, dtype=int) if img_cond_steps > 1 else [end]
    force_indices = np.linspace(start, end, force_cond_steps, dtype=int) if force_cond_steps > 1 else [end]
    cond_indices = np.linspace(start, end, cond_steps, dtype=int) if cond_steps > 1 else [end]
    np_images = np.array([ get_image(os.path.join(image_dir, f'{i:06d}.png')) for i in image_indices] ) # TxHxWxC
    data_dict = {'image': torch.from_numpy(np_images).float().to(device)}
    states = ep_states
    for k in obs_fields:
        obs = states[k]; obs_index = cond_indices if 'force' not in k else force_indices
        obs = np.array([obs[i] for i in obs_index])
        data_dict[k] = torch.from_numpy(obs).float().to(device)
    return data_dict

def get_score(data, model):
    image = data["image"].to(device); force = data["force"].to(device)
    gripper = data["gripper_width"].unsqueeze(-1)
    pose = torch.cat([data["pose"], gripper], dim=-1).to(device)
    return nn.functional.sigmoid( model(image, pose, force)['logits'] )

def main():
    ep_ids = [39]
    dir = '/home/atkesonlab4/Desktop/YiqiProject/100%_Project/ethernet-plugging/logs-collectfailures/75rl-rtc'
    model = get_model()

    for ep_id in ep_ids:
        ep_dir = os.path.join(dir, f"episode{ep_id:06d}")
        states = np.load(os.path.join(ep_dir, 'states.npz'))
        logit_path = os.path.join(ep_dir, 'fail_detect.npy')
        mp4_path = os.path.join(ep_dir, 'fail_detect.mp4')
        N = len( states['pose'] )
        data_dict = None 
        for j in tqdm(range(N)):
            if data_dict is None:
                data_dict = { k: v.unsqueeze(0) for k,v in get_data(ep_dir, j, states ).items() }
            else:
                _data_dict = get_data(ep_dir, j, states )
                for k,v in data_dict.items():
                    data_dict[k] = torch.cat([ v, _data_dict[k].unsqueeze(0)], dim = 0)
        with torch.no_grad():
            scores = get_score(data_dict, model).cpu().numpy() 
            np.save( logit_path, scores)
        
        # Get all the images 
        images = [np.array(Image.open(os.path.join(ep_dir, 'images', f"{j:06d}.png"))) for j in range(N)]
        save_video_with_running_score(images, scores, mp4_path)
        

if __name__ == "__main__":
    main()