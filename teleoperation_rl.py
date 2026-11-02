from agent.eval.eval_realtime import EvalRealtimeChunking
from robot.env import GRIP_OPEN, GRIP_CLOSED
import numpy as np
from robot import fdcc
import argparse
import time
import os

STIFF_Z_K = 3000.0      # [N/m] on base z while Cross is toggled on, to push the plug in (as teleoperation.py)
STIFF_Z_RAMP_S = 0.1    # blend in/out over this long (fdcc.toml gain_ramp_s)


class TeleoperationRL(EvalRealtimeChunking):
    # No init-- inherited from EvalRealtimeChunking.
    GRIP_OVERRIDE_AGREE_S = 0.3     # policy must match a gripper override this long to take back over

    def pre_reset(self):
        print('============ EVAL+TELEOPERATION ============')
        print('Control inputs (above) add offsets to base policy.')
        print('Circle overrides the gripper: it flips the commanded open/close and holds it')
        print('until the policy has predicted the same state for '
              f'{self.GRIP_OVERRIDE_AGREE_S:.1f} s (press again to flip back).')
        print(f'Cross toggles base z stiffness to {STIFF_Z_K:.0f} N/m, overriding the policy\'s.')
        # print('Dpad-Left rewinds actions.')
        # print('- press = rewind 1s')
        # print('- hold = rewind until released')
        print('============================================')
        print()

    grip_override = None            # GRIP_OPEN / GRIP_CLOSED while the operator holds the gripper
    _grip_agree_t = None            # since when the policy has agreed with the override
    stiff_z = False                 # Cross: base z K overridden to STIFF_Z_K
    _stiff_w = 0.                   # blend weight of the override, ramps toward stiff_z

    # undo_action_buffer = []
    # action_hist = []
    def get_action(self):
        # Rewind (Dpad-Left) disabled: replaying old commands left the command low-pass on its
        # pre-rewind state, and the rewound stretch would be recorded as demonstration.
        # if self.iface.dualsense.state.DpadLeft:
        #     # Interruption signal - Undo last 1s, hold for longer.
        #     if len(self.undo_action_buffer) == 0:
        #         undo_count = int(self.control_freq)
        #         self.undo_action_buffer = self.action_hist[-undo_count:]
        #         self.action_hist = self.action_hist[:-undo_count]
        #     elif len(self.undo_action_buffer) == 1 and len(self.action_hist) > 0:
        #         # If undo about to finish but we want more, add 1 at a time.
        #         next_act = self.action_hist.pop()
        #         self.undo_action_buffer.insert(0, next_act)
        #
        # if len(self.undo_action_buffer) > 0:
        #     action = self.undo_action_buffer.pop()
        #     if len(self.undo_action_buffer) == 0:
        #         # Finished undo-ing actions, empty prediction buffer
        #         self.buffer.clear()
        #     return action

        # super() = RealtimeChunking.get_action()
        # gets nn prediction from RealtimeChunkingBuffer
        if self.iface.act.get('toggle_stiff_z'):
            self.toggle_stiff_z()
        nn_action = super().get_action()
        des_pose, des_grip, _, _ = self._unshortcut_action(nn_action)
        des_pose = self.iface.residual_action(np.array(des_pose), self.control_dt)
        des_grip = self.override_gripper(des_grip)
        # self.action_hist.append((des_pose, des_grip))
        return des_pose, des_grip

    def toggle_stiff_z(self):
        """
        Cross: base z stiffness to STIFF_Z_K; again: back to the policy's (or fdcc.toml's).
        With a stiffness head the override is applied to every predicted command (apply_gains),
        blended in log K over STIFF_Z_RAMP_S; without one it is set once, ramped by Env.
        """
        self.stiff_z = not self.stiff_z
        print(f'\nstiff z {"ON" if self.stiff_z else "off"}')
        if 'stiffness' in self.policy.impedance_fields:
            return
        if self.stiff_z:
            K = fdcc._six(self.env.imp.p.K)
            K[2] = STIFF_Z_K
            self.env.set_gains(K=K, frame='base')
        else:
            self.env.restore_gains()

    def apply_gains(self, log_gains):
        step = self.control_dt / STIFF_Z_RAMP_S
        self._stiff_w = float(np.clip(self._stiff_w + (step if self.stiff_z else -step), 0., 1.))
        if self._stiff_w > 0 and 'stiffness' in log_gains:
            K = np.array(log_gains['stiffness'], float)
            w = 3 * self._stiff_w ** 2 - 2 * self._stiff_w ** 3     # smoothstep, as fdcc ramps
            K[2] += w * (np.log10(STIFF_Z_K) - K[2])
            log_gains = {**log_gains, 'stiffness': K}
        super().apply_gains(log_gains)

    def override_gripper(self, policy_grip):
        """
        Circle flips the executed gripper state and latches it, to break a policy that keeps
        putting a grasp / release off (a stationary scene predicts the same "in 0.5 s" every
        chunk). The policy gets it back once its own prediction has matched for
        GRIP_OVERRIDE_AGREE_S. The commanded (overridden) state is what rawdata.h5 records.
        """
        now = time.time()
        if self.iface.act.get('right_gripper'):
            executed = self.last_action[1]
            self.grip_override = GRIP_OPEN if executed == GRIP_CLOSED else GRIP_CLOSED
            self._grip_agree_t = None
            print(f'\ngripper override: {"close" if self.grip_override == GRIP_CLOSED else "open"}')
        if self.grip_override is None:
            return policy_grip
        if policy_grip == self.grip_override:
            if self._grip_agree_t is None:
                self._grip_agree_t = now
            elif now - self._grip_agree_t >= self.GRIP_OVERRIDE_AGREE_S:
                print('\ngripper override released: policy agrees')
                self.grip_override = self._grip_agree_t = None
                return policy_grip
        else:
            self._grip_agree_t = None
        return self.grip_override


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Teleoperation script for Ethernet Plugging task')
    parser.add_argument('--ckpt', type=str, required=True, help='path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str,
                        default='/home/atkesonlab4/Desktop/YiqiProject/100%_Project/dataset/ethernet_plugin_unplug_rl2',
                        help='Base dataset directory')
    parser.add_argument('--control_freq', '--hz', type=float, default=100.0,
                        help='command frequency (Hz) of the real-time loop; chunks are interpolated in '
                             "time, so it need not match the policy's framerate")
    parser.add_argument('--weight_decay', type=float, default=0.5,
                        help='recency-weighting rate (1/s) for ensembling overlapping chunks')
    parser.add_argument('--taper', type=float, default=0.15,
                        help='seconds over which a chunk\'s weight ramps in after it arrives and out '
                             'before its horizon ends (0 = off)')
    parser.add_argument('--lpf', type=float, default=3.0,
                        help='cutoff (Hz) of the low-pass on the commanded pose, delay-compensated (0 = off)')
    parser.add_argument('-d', '--debug', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if not args.debug:
        indices = [
            int(d.removeprefix('episode'))
            for d in os.listdir(args.log_dir)
            if d.startswith('episode') and d.removeprefix('episode').isdigit()
        ] if os.path.exists(args.log_dir) else []
        args.id = max(indices, default=0) + 1
        print(f'Auto-selected episode ID: {args.id}')
        print(f"Saving data to: {args.log_dir}, Episode {args.id}")

        os.makedirs(args.log_dir, exist_ok=True)
        path = args.log_dir
    else:
        path = None
    teleop = TeleoperationRL(
        ckpt=args.ckpt, device=args.device,
        log_dir=path,
        control_freq=args.control_freq, weight_decay=args.weight_decay,
        taper=args.taper, lpf_hz=args.lpf,
        env_metadata={'id': args.id} if path is not None else None,
    )
    teleop.run()
