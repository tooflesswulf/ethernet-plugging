from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pathlib
import h5py


def ewma(data, alpha=.05):
    acc = 0
    vals = []
    for di in data:
        acc = alpha * di + (1 - alpha) * acc
        vals.append(acc)
    return np.array(vals)


def main(args):
    with h5py.File(args.file) as f:
        gobs = f['gripper_obs']
        gdat = {k: np.array(gobs[k]) for k in gobs.keys()}

        robs = f['robot_obs']
        rdat = {k: np.array(robs[k]) for k in robs.keys()}

        cobs = f['camera_obs']
        cdat = {k: np.array(cobs[k]) for k in cobs.keys()}

    smoothed = ewma(rdat['actual_force'][:, 2], alpha=args.alpha)
    delta = args.tx
    delt_r = int(delta * 500)  # Nominal 500Hz
    delt_g = int(delta * 250)  # Nominal 250Hz
    cam_dt_ms = 1000 * np.median(np.diff(cdat['time']))

    frames = list(range(1, len(cdat['time'])))
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.2], hspace=0.4)
    ax_img = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[2, 1])

    ax_img.set_aspect('equal')
    ax_img.axis('off')

    # Initial frame
    i0 = frames[0]
    t0 = cdat['time'][i0]
    ri0 = np.abs(rdat['time'] - t0).argmin()
    gi0 = np.abs(gdat['time'] - t0).argmin()
    img_artist = ax_img.imshow(cdat['image_bgr'][i0][:, :, ::-1])
    ax_img.set_title(f'Wrist camera')

    line_fraw, = ax1.plot(rdat['time'][ri0 - delt_r:ri0], rdat['actual_force'][ri0 - delt_r:ri0, 2])
    line_fsmooth, = ax1.plot(rdat['time'][ri0 - delt_r:ri0], smoothed[ri0 - delt_r:ri0])
    ax1.legend(['eef force', 'smoothed'], loc=1)

    line_gf, = ax2.plot(gdat['time'][gi0 - delt_g:gi0], gdat['gripper_force'][gi0 - delt_g:gi0], 'C2')
    ax2.legend(['gripper force'], loc=1)

    line_gw, = ax3.plot(gdat['time'][gi0 - delt_g:gi0], gdat['gripper_width'][gi0 - delt_g:gi0], 'C2')
    ax3.legend(['gripper width'], loc=1)

    plt.close(fig)

    def make_update(frames, img_artist, line_fraw, line_fsmooth, line_gf, line_gw, ax1, ax2, ax3):
        def update(frame_idx):
            i = frames[frame_idx]
            t = cdat['time'][i]
            ri = np.abs(rdat['time'] - t).argmin()
            gi = np.abs(gdat['time'] - t).argmin()

            img_artist.set_data(cdat['image_bgr'][i][:, :, ::-1])

            r_sl = slice(max(0, ri - delt_r), ri)
            line_fraw.set_data(rdat['time'][r_sl], rdat['actual_force'][r_sl, 2])
            line_fsmooth.set_data(rdat['time'][r_sl], smoothed[r_sl])
            ax1.set_xlim(rdat['time'][r_sl.start], rdat['time'][max(r_sl.stop - 1, r_sl.start)])
            ax1.relim()
            ax1.autoscale_view(scalex=False)

            g_sl = slice(max(0, gi - delt_g), gi)
            line_gf.set_data(gdat['time'][g_sl], gdat['gripper_force'][g_sl])
            ax2.set_xlim(gdat['time'][g_sl.start], gdat['time'][max(g_sl.stop - 1, g_sl.start)])
            ax2.relim()
            ax2.autoscale_view(scalex=False)

            line_gw.set_data(gdat['time'][g_sl], gdat['gripper_width'][g_sl])
            ax3.set_xlim(gdat['time'][g_sl.start], gdat['time'][max(g_sl.stop - 1, g_sl.start)])
            ax3.relim()
            ax3.autoscale_view(scalex=False)

            return img_artist, line_fraw, line_fsmooth, line_gf, line_gw
        return update

    updator = make_update(frames, img_artist, line_fraw, line_fsmooth, line_gf, line_gw, ax1, ax2, ax3)
    anim = FuncAnimation(fig, updator, frames=len(frames),
                         interval=cam_dt_ms, blit=True)

    anim.save(pathlib.Path(args.file).parent / 'video.mp4', fps=30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True,
                        help='Path to the .h5 file containing the data to visualize.')
    parser.add_argument('--tx', type=int, default=5,
                        help='Time scale (s) of x-axis in the plots.')
    parser.add_argument('-a', '--alpha', type=float, default=0.02,
                        help='Smoothing factor for the EWMA of the force signal.')
    args = parser.parse_args()

    main(args)
