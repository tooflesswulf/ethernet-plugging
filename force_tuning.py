import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import robot_execution
from env import URPose

GRIP_WIDTH_MM = 10
GRIP_FORCE_N = 40
GRIP_SPEED_MMPS = 50
GRIP_PULLBACK_MM = 5

PLOT_WINDOW_S = 10.0    # seconds of force history shown in the live plot
PLOT_REFRESH_HZ = 15.0  # plot redraw rate, independent of control_freq

FORCE_LABELS = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
FORCE_UNITS = ['N', 'N', 'N', 'Nm', 'Nm', 'Nm']
FZ_IDX = FORCE_LABELS.index('Fz')


class ForceTuner(robot_execution.RobotExecution):
    """Teleoperation without data collection: shows a live plot of
    `Env.robot_obs[-1].actual_force` and exposes sliders to tune the force
    exponential filter alpha and the adaptive z-force PID gains (kp, kd).
    """

    @staticmethod
    def add_args(parser):
        parser.add_argument('--window', type=float, default=PLOT_WINDOW_S,
                            help='Seconds of force history shown in the live plot')

    def pre_reset(self):
        print('============ FORCE TUNING (no data collection) ============')
        print('Triangle toggles adaptive z-force mode, which uses kp/kd below.')
        print('Drag the sliders to tune force_alpha / kp / kd live.')
        print('=============================================================')

    def __init__(self, args):
        control_freq = 100
        home_pose = URPose(-0.147, 0.612, 0.184, 2.44, 2.44, 0.633)  # low-position (cable easy to see)
        self.plot_window_s = args.window

        super().__init__(home_pose=home_pose, control_freq=control_freq,
                         gforce=GRIP_FORCE_N, gwidth=GRIP_WIDTH_MM,
                         gspeed=GRIP_SPEED_MMPS, gpullback=GRIP_PULLBACK_MM,
                         env_metadata={}, show_image=True,
                         path=None)
        self._setup_plot()

    # ------------------------------------------------------------------
    # Live plot + tuning GUI
    # ------------------------------------------------------------------
    def _setup_plot(self):
        plt.ion()
        self.fig = plt.figure('Force Tuning', figsize=(12, 9))
        gs = self.fig.add_gridspec(2, 3, left=0.08, right=0.97, top=0.93, bottom=0.40,
                                   hspace=0.45, wspace=0.3)

        self.axes = [self.fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]
        self.lines = []
        for i, (ax, label, unit) in enumerate(zip(self.axes, FORCE_LABELS, FORCE_UNITS)):
            line, = ax.plot([], [], linewidth=1, label='raw')
            self.lines.append(line)
            ax.set_xlim(-self.plot_window_s, 0)
            ax.set_title(label)
            ax.grid(True)
            if i % 3 == 0:
                ax.set_ylabel(unit)
            if i // 3 == 1:
                ax.set_xlabel('Time (s)')

        self.filt_line, = self.axes[FZ_IDX].plot([], [], color='k', linewidth=2, label='filtered')
        self.axes[FZ_IDX].legend(loc='upper right', fontsize=7)
        self.fig.suptitle('robot_obs actual_force')

        ax_alpha = self.fig.add_axes([0.15, 0.24, 0.7, 0.03])
        ax_kp = self.fig.add_axes([0.15, 0.16, 0.7, 0.03])
        ax_kd = self.fig.add_axes([0.15, 0.08, 0.7, 0.03])

        # kp/kd/alpha are tuned across orders of magnitude (e.g. kd ~ 1e-5),
        # so use log-scaled sliders: the handle position tracks log10(value)
        # and we override valtext to display the real (non-log) value.
        self.slider_alpha = self._add_log_slider(
            ax_alpha, 'force_alpha (log)', 1e-3, 1.0, self.env.force_alpha,
            '{:.4f}', lambda v: setattr(self.env, 'force_alpha', v))
        self.slider_kp = self._add_log_slider(
            ax_kp, 'kp (log)', 1e-5, 1e-1, self.env.kp,
            '{:.2e}', lambda v: setattr(self.env, 'kp', v))
        self.slider_kd = self._add_log_slider(
            ax_kd, 'kd (log)', 1e-7, 1e-3, self.env.kd,
            '{:.2e}', lambda v: setattr(self.env, 'kd', v))

        plt.show(block=False)
        self._last_plot_t = 0.0
        self._plot_period = 1.0 / PLOT_REFRESH_HZ

    @staticmethod
    def _add_log_slider(ax, label, vmin, vmax, valinit, fmt, setter):
        """Slider whose handle position tracks log10(value), so dragging it
        gives fine control near small magnitudes (e.g. kd ~ 1e-5) instead of
        squeezing them into a sliver at one end of a linear range.
        """
        valinit = np.clip(valinit, vmin, vmax)
        slider = Slider(ax, label, np.log10(vmin), np.log10(vmax),
                        valinit=np.log10(valinit), valfmt='%1.2f')
        slider.valtext.set_text(fmt.format(valinit))

        def _on_changed(log_v):
            val = 10 ** log_v
            setter(val)
            slider.valtext.set_text(fmt.format(val))

        slider.on_changed(_on_changed)
        return slider

    def _figure_open(self):
        return plt.fignum_exists(self.fig.number)

    def _update_plot(self):
        servo_hz = self.env.servo_frequency
        n = int(self.plot_window_s * servo_hz) + 10
        recent = self.env.robot_obs[-n:]
        if not recent:
            return

        t_latest = recent[-1].time
        times = np.array([o.time for o in recent]) - t_latest
        forces = np.array([o.actual_force for o in recent])
        filt_fz = np.array([o.filtered_force.z for o in recent])

        keep = times >= -self.plot_window_s
        times, forces, filt_fz = times[keep], forces[keep], filt_fz[keep]
        if len(times) == 0:
            return

        for i, (ax, line) in enumerate(zip(self.axes, self.lines)):
            channel = forces[:, i]
            line.set_data(times, channel)
            ymax = max(1.0, float(np.abs(channel).max()) * 1.1)
            ax.set_ylim(-ymax, ymax)
            ax.set_xlim(-self.plot_window_s, 0)
        self.filt_line.set_data(times, filt_fz)

    def post_step(self, obs, action):
        if not self._figure_open():
            return
        now = time.perf_counter()
        if now - self._last_plot_t >= self._plot_period:
            self._last_plot_t = now
            self._update_plot()
        plt.pause(0.001)

    def runtime_info(self):
        obs = self.last_obs
        fz = obs['state']['filtered_force'].z
        mode = 'ON ' if self.env.adaptive_mode else 'OFF'
        print(f"adaptive={mode} | alpha={self.env.force_alpha:.4f} "
             f"kp={self.env.kp:.2e} kd={self.env.kd:.2e} | "
             f"fz={fz:6.2f} des_zforce={self.env.des_zforce:6.2f}", end='\r')

    def close(self):
        super().close()
        if self._figure_open():
            plt.close(self.fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Live force plot + filter/PID tuning GUI for the Ethernet Plugging task (no data collection)')
    ForceTuner.add_args(parser)
    args = parser.parse_args()

    tuner = ForceTuner(args)
    tuner.run()
