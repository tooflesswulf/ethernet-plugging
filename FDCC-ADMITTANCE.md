# FDCC-style Cartesian admittance over speedL (UR16e)

This is a spec for rebuilding the controller from scratch. The controller's numbers
live in [`fdcc.toml`](fdcc.toml) and this file uses their names; teleop shaping values
are given inline, and the measurements behind the numbers are in §11. Poses are elements of
$SE(3)$, twists of $\mathfrak{se}(3)$, and wrenches of $\mathfrak{se}(3)^*$.
The reference implementation, `test-fdcc-admittance.py`, is a first-order
approximation of what's written here; §6 lists the differences.

## 1. What it is

A 500 Hz host-side loop around UR's `speedL` velocity interface:

```
target T_t, V_t (teleop)
        |
W (getActualTCPForce) -> F in se(3)* at c -> [ M dV/dt + D (V - g V_t) = F + K xi ] -> V in se(3) at c -> Ad -> speedL
T (getActualTCPPose) ---> xi = log(T_c^-1 T_t)                                     (clamped: speed, accel)
```

- **We own the geometry.** The compliance frame $c$ sits wherever `compliance_point`
  says. forceMode's is stuck at the flange.
- **UR's joint velocity loops sit underneath**, so joint stiction (12–22 Nm on joint 1)
  is theirs to absorb. That stiction is what killed the direct-torque version.
- **The spring uses the measured pose; $V$ is the controller's own state.** The measured
  twist is noisy and lags.

## 2. Notation

**Frames:**
- $s$: robot base. The UR `base` frame, not ROS `base_link`; the two differ by $R_z(\pi)$.
- $e$: the TCP.
- $f$: the flange.
- $c$: the compliance frame, with the TCP's axes and origin at `compliance_point`.
- $c^\star$: the target compliance frame.

$T_{ab}\in SE(3)$ is the pose of frame $b$ in frame $a$, written $(R_{ab}, p_{ab})$.

$$
T_{ec} = \big(I,\ r_c\big)\ \ (r_c = \texttt{compliance\_point}),\qquad
T_{fe} = \texttt{tcp\_offset},\qquad
T_{sc} = T_{se}T_{ec},\qquad
T_{sf} = T_{se}T_{fe}^{-1}.
$$

**Twists and wrenches.** Coordinates are ordered linear-first, as UR does:
$V = [v;\ \omega]\in\mathfrak{se}(3)$ and $F = [f;\ \tau]\in\mathfrak{se}(3)^*$. They pair
through the power $\langle F, V\rangle = f\cdot v + \tau\cdot\omega$.

$$
\hat V = \begin{bmatrix}[\omega]_\times & v\\ 0 & 0\end{bmatrix},\qquad
\mathrm{Ad}_{T} = \begin{bmatrix}R & [p]_\times R\\ 0 & R\end{bmatrix},
$$

$$
V_a = \mathrm{Ad}_{T_{ab}}\,V_b,\qquad
F_b = \mathrm{Ad}_{T_{ab}}^{\top}\,F_a,\qquad
\langle F_a, V_a\rangle = \langle F_b, V_b\rangle .
$$

**Body twist:** $V^b_c = (T_{sc}^{-1}\dot T_{sc})^\vee$. Every controller quantity is a
body quantity at $c$: a twist or wrench **at** the origin of $c$, **in** the axes of $c$.

**UR's "mixed" quantities.** RTDE reports twists and wrenches with base-frame axes but
at a body point. With $B(R) = \mathrm{diag}(R, R)$:

| RTDE call | what it is |
|---|---|
| `getActualTCPPose` | $T_{se}$ as $[p;\ \mathrm{rotvec}(R)]$ |
| `getActualTCPSpeed`, `speedL` argument | $B(R_{se})\,V^b_e$ (TCP point, base axes) |
| `getActualTCPForce` | $W = B(R_{sf})\,F^b_f$ (moment about the **flange**, base axes; measured with §9.1, UR's docs say TCP) |

## 3. The per-cycle algorithm

Run everything below once per 2 ms cycle, between `initPeriod()` and `waitPeriod()`.

**0. Read the robot and check the aborts.**
- Read $T_{se}$ and $W$, and the stop flags.
- Abort on a protective or emergency stop.
- Abort if $\|W_f\| > $ `abort_wrench[0]`, $\|W_\tau\| > $ `abort_wrench[1]`, or
  $\|p_{se} - p_{se}(0)\| > $ `max_drift_m`.

**1. Wrench at c.** Convert $W$ to a body wrench at $c$, then filter, deadband and clamp it:

$$
F = \operatorname{clamp}\circ\operatorname{db}\circ\operatorname{LPF}
\Big(\mathrm{Ad}_{T_{fc}}^{\top}\,B(R_{sf})^{\top}\,\sigma W\Big),
\qquad T_{fc} = T_{sf}^{-1}T_{sc},\ \ \sigma = \texttt{sign}.
$$

$\operatorname{LPF}$ is a 30 Hz first-order filter per component. The deadband $\operatorname{db}$ acts on the norms
of the two halves: $f \to f\,\max(\|f\| - 1.5, 0)/\|f\|$, and likewise $\tau$ with 0.15.
The clamp limits the norms of the halves to `clamp`.

**2. Error and target twist.** Both are body quantities at $c$:

$$
\xi = \log\!\big(T_{sc}^{-1}\,T_{sc^\star}\big)^\vee \in \mathfrak{se}(3),
\qquad
V_t = \mathrm{Ad}_{T_{cc^\star}}\,V^b_{c^\star},\quad T_{cc^\star} = \exp(\hat\xi).
$$

$\xi$ points from the current pose to the target. Its linear part is
$V^{-1}(\omega)\,R_{sc}^\top(p_{sc^\star}-p_{sc})$, where
$V^{-1}(\omega) = I - \tfrac12[\omega]_\times + O(\omega^2)$. The target's own body twist
$V^b_{c^\star}$ comes from teleop (§4).

**3. Feedforward fade.** Do the linear and angular halves separately, with $f_r$ = `release`
and $T_r$ = `recover_s`:

$$
g^{+} = \min\!\Big(\operatorname{clip}\big(1 + \tfrac{\langle F_{\text{half}},\, V_{t,\text{half}}\rangle}{\|V_{t,\text{half}}\|\,f_r},\,0,\,1\big),\ g + \tfrac{\Delta t}{T_r}\Big).
$$

A negative pairing means resistance. The fade drops instantly and recovers slowly.
$F$ in this pairing is a 5 Hz low-passed copy (`filter_hz`), not the 30 Hz one the
dynamics use. With the fast one, the ~28 Hz force wobble after an impact kept the fade
tripped, and the lost feedforward left a steady lag that kept the wobble going.

**4. Dynamics.** $M, D, K$ are diagonal in the body coordinates of $c$, and $s$ is the
stiffness-button scale:

$$
M\dot V + D\,(V - g V_t) = F + K_s\,\xi,\qquad K_s = sK,\ \ D_s = \sqrt{s}\,D .
$$

Integrate it with the damping term implicit. That's stable for any $D \ge 0$ and still
defined at $M = 0$:

$$
V^{+} = \big(M + D_s\Delta t\big)^{-1}\big(M V + \Delta t\,(F + K_s\xi + D_s\,g V_t)\big).
$$

On stiff axes (`selection` = 0) the force is ignored and the axis tracks:
$V_i = V_{t,i} + k_s\,\xi_i$, with $k_s$ = `stiff_gain`.
$\dot V$ is taken componentwise. The rigid-body Coriolis term $-\mathrm{ad}_V^{*}MV$ is
deliberately left out: this is a virtual system, not a body.

**5. Clamps and the command.** The state itself is clamped, so it can't wind up behind the clamps:

$$
V^{+} \leftarrow \operatorname{clamp}_{\|\cdot\|}\big(V^{+},\ \texttt{speed}\big),\qquad
V \leftarrow V + \operatorname{clamp}_{\|\cdot\|}\big(V^{+} - V,\ \texttt{accel}\,\Delta t\big),
$$

$$
\texttt{speedL}\Big(\operatorname{clamp}_{\|\cdot\|}\big(B(R_{se})\,\mathrm{Ad}_{T_{ec}}V,\ \texttt{speed}\big),\ a = \texttt{accel}[0],\ t = \Delta t\Big).
$$

$\operatorname{clamp}_{\|\cdot\|}(x, [a, b])$ scales the linear half to norm $\le a$ and the
angular half to norm $\le b$, keeping direction. The linear half depends on the point,
so the command is clamped again at the TCP.

**Rotating K to base axes (optional).** To have fixed-axis stiffness, i.e.
`frame = "base"` with $K_s$ diagonal in base axes, use
$K = B(R_{sc})^\top K_s B(R_{sc})$, and the same for $D$ and $M$. This makes them
pose-dependent.

## 4. Teleop target

Update the target $T_{sc^\star}$ at the teleop rate (100 Hz, $h$ = 10 ms). Hold $V^b_{c^\star}$
between updates.

1. **Stick twist.** Let the DualSense move a copy of the target to $T_{\text{stick}}$.
   Then $V_{\text{stick}} = \log(T_{sc^\star}^{-1}T_{\text{stick}})^\vee/h$, a body twist.
   `interface.py` composes rotations on the left, in base axes; the log turns that into a body twist.
2. **Slew limit.**
   $V_{\text{slew}} \leftarrow V_{\text{slew}} + \operatorname{clamp}_{\|\cdot\|}(V_{\text{stick}} - V_{\text{slew}},\ a_{\text{stick}}\,h)$,
   with $a_{\text{stick}}$ = 0.5 m/s², 2 rad/s² (tested),
   then $T_{\text{new}} = T_{sc^\star}\exp(h\hat V_{\text{slew}})$.
3. **Non-dragging leash**, limit in newtons: the leash distance is
   `leash_N / K` per half (largest K of the half), so the spring can push at most
   `leash_N`; in contact that is the only push left once the feedforward fades. A fixed
   30 mm at K 300 capped insertion at ~9 N. (`env.py` implements this as a target that
   chases `des_pose` under that limit, `fdcc.leash_step`.)
   Let $\xi_{\text{new}} = \log(T_{sc}^{-1}T_{\text{new}})^\vee$.
   If the linear half has norm above `leash[0]` **and** larger than before this update,
   undo that half of the step and zero that half of $V_{\text{slew}}$. Do the same for
   the angular half with `leash[1]`. Never move the target toward the arm.
4. **Commit and feed forward.** $T_{sc^\star} \leftarrow T_{\text{new}}$, and
   $V^b_{c^\star} = \log(T_{\text{old}}^{-1}T_{sc^\star})^\vee/h$. This is only the
   stick's own motion, never the arm's.
5. **Buttons.**
   - Dpad Up/Down: $s \times 1.5$ or $s / 1.5$.
   - Cross: $T_{sc^\star} = T_{sc}$ and $V_{\text{slew}} = 0$. A jump, not a velocity.
   - Square: rotation lock. Angular `selection` → 0; set the target's rotation to the
     current rotation. Also a jump.
   - Dpad Left: stop, then zero the F/T (out of contact only).

## 5. Special cases of the same law

**(a) Pose target:** use §3 as written.
- Free-space steady state: $V = V_t$, and $\xi = -K^{-1}F$.
- Against contact with a still target: $F = -K\xi$, bounded by $K\cdot$`leash` in teleop.

**(b) Velocity target, not anchored:** $K = 0$, so $M\dot V + D(V - gV_t) = F$.
- Nothing pulls it back, so force errors make it drift. That's why teleop integrates
  the stick into a pose target and uses (a).
- In static contact against $V_t$, the resisting force per half is
  $D\|V_t\|f_r/(f_r + D\|V_t\|) < f_r$. It was $D\|V_t\|$ without the fade: 27 N at 27 mm/s.

**(c) Force target** $F_d \in \mathfrak{se}(3)^*$, the wrench the tool should apply:
$M\dot V + DV = F + F_d$.
- Static contact gives $F = -F_d$, with no integrator needed.
- In free space it approaches at $V = D^{-1}F_d$.

**Hybrid,** with $S_f$ selecting force axes and $S_p = I - S_f$ selecting motion axes:

$$
M\dot V + D\,(V - S_p\,g\,V_t) = F + S_f F_d + S_p K\xi .
$$

## 6. How `test-fdcc-admittance.py` differs

The script uses a decoupled error in mixed coordinates. Within the leash it's the same
controller, to first order:

| here | script | difference |
|---|---|---|
| $\xi = \log(T_{sc}^{-1}T_{sc^\star})^\vee$ | $e = [p_c - p_{c^\star};\ \mathrm{rotvec}(R R_{\star}^\top)]$, rotated to $c$ axes | sign; the linear part lacks $V^{-1}(\omega)$. At the leash limits (0.25 rad, 30 mm) that's up to ~3.75 mm, sideways |
| $V$ stored as a body twist at $c$ | stored in base axes, rotated in and out each cycle | a rotating-frame term, second order |
| $V_t = \mathrm{Ad}_{T_{cc^\star}}V^b_{c^\star}$ | finite-difference target twist, moved from point $c^\star$ to point $c$ | same, to first order |
| $F$ via $\mathrm{Ad}^\top$ | $\tau_c = \tau_f + (p_f - p_c)\times f$, in base axes, then rotated | identical |

Avoid $\log(T_{sc^\star}T_{sc}^{-1})$, the spatial error. Its linear part is the
velocity of the point at the base origin, which puts the compliance centre there,
0.5–1 m away.

## 7. Start, stop and shutdown (order matters)

**Start-up:**
1. Connect with `RTDEControlInterface(ip, 500)`.
2. Read `getPayload()`, `getPayloadCog()` and `getTCPOffset()`. Log them, and refuse
   to run if they're outside the tolerances.
3. Do all slow setup: DualSense, `import interface`, and so on. This is seconds of silence.
4. Zero the F/T: stop, wait 0.3 s, `zeroFtSensor()`, wait 0.3 s. Out of contact only.
5. `setWatchdog(watchdog_hz)`. From here on the host must never be silent for more
   than 50 ms.
6. Run the loop.

**Stop** (every time, mid-run as well):
1. Ramp the last commanded twist to zero with `speedL` at `accel`, every cycle,
   until the measured speed is below `rest_speed` (at most 0.5 s).
2. Then `speedStop(stop_decel)`.
3. Reset $V$ and the filter.

**Shutdown:** in a `finally` block: ramp down, then `speedStop`, then `stopScript()`.

**While the watchdog is armed:**
- Never call a blocking command. That includes `moveL` and a `speedStop` issued
  while the arm is still moving.
- Wait in steps of 10 ms or less, calling `kickWatchdog()` each time. That includes
  keyboard prompts (poll stdin with `select`).
- Return to a pose by running the same loop with every axis stiff.

## 8. Pitfalls found on the hardware (don't repeat these)

| Symptom | Cause | Rule |
|---|---|---|
| Stalled host → arm keeps moving | ur_rtde `speed_thread` re-issues the last `speedl` forever | `time` is not a stall guard; use `setWatchdog` |
| 125 Hz buzz whenever the speed changes | `speedl(t = 8 ms)`: the script takes a new target only when it returns → 8 ms staircase | `time` = 1 control cycle |
| C207A0 "Fieldbus input disconnected" | watchdog tripped: `speedStop` from speed (the script's `stopl` blocks), `moveL`, slow imports after arming | §7 |
| Watchdog trip = **protective** stop | that's how this controller implements the watchdog's "stop" | clear it on the pendant |
| `getActualTCPForce` exactly 0 on every axis | external F/T input left enabled (a probe died before cleanup) | `ftRtdeInputEnable(False)` once. Never enable it: that protective-stops |
| Arm kept pressing into a wall (43 N, abort) | feedforward $D V_t$ = 27 N at 27 mm/s, larger than the 25 N clamp | fade the feedforward (§3, step 3) |
| (suspected) slow to back off an impact | accel clamp 0.5 m/s²; only seen together with the feedforward bug above | `accel = 2`, clean in the tap test |
| 25 Hz limit cycle, 9 N rms, with a steady force on | getActualTCPForce reads the arm's own accel as ~20 kg, 34 ms late; M 10 kg → loop gain 1.55 | $M \ge$ ~15 kg |
| 25 Hz ringing at 40–50 mm/s | symmetric fade: force wobble × $\|V_t\|/f_r$ = 17 mm/s per N | fast-attack / slow-release fade |
| Rotation ringing at 25 Hz | lateral force × 154 mm flange lever → torque ($\mathrm{Ad}^\top$ from $f$ to $c$); I = 0.3 kg m² too light | $I = 0.6$ |
| Fade trips at every stick start | the arm's own acceleration reads as ~5 N of resistance | slew-limit the stick (0.5 m/s², 2 rad/s²) |
| False contact at tap start | same phantom force on a step start | ramp the approach (0.3 s) |
| Hand push moved the equilibrium; arm ran to 80 mm/s | leash dragged the target after the arm, and the dragged velocity went into the feedforward | non-dragging leash; $V_t$ only from the stick |
| Adaptive z-force (Triangle) made the arm go wild | old servoL PID set target z = actual z + kp·err + kd·d(err)/dt: a target that moves with the arm feeds forward, and the kd term turned force noise into target jumps | integrate the force error into the target, rate-limited (`[zforce]`) |
| Scripted moves stall ~2 mm short of `move_to` tolerance | at teleop K the last mm close with D/K = 3.3 s (2 s rotation) | stiffen during scripted moves (`[scripted]`, via `env.set_gains`) |
| Lurch at scripted-move starts/ends | a K step changes the spring force by ΔK·ξ at once; constant-speed scripted targets also start/stop the feedforward in one step | bumpless gain change (rescale ξ by K_old/K_new, `Impedance.bumpless_target`); minimum-jerk `MotionStep` profile |
| Force "saturates" when pushing by hand | 25 N / 2 Nm input clamp | open: raise toward the abort limits if wanted |

The M trade-off: lighter reacts to contact better but gives the 25 Hz loop through
the phantom mass more gain. At 10 kg it limit-cycled; 15 kg was clean in the tap
test. Contact at masses of 17 kg and up has not been checked on the arm.

## 9. Validation tests

1. **Compliance centre (spin ratio).** Set $K = 0$, everything compliant, deadband 0.
   Push sideways at frames $d$ offset along tool $z$ (TCP and flange). The push-point
   twist is $V_d = \mathrm{Ad}_{T_{ed}}^{-1}V^b_e$. Per point, take
   $s = \operatorname{median}(\|\omega\|/\|v_d\|)$ over samples with $\|f\| > 3$ N,
   leaving out samples where a speed clamp is active. Fit
   $s(z_d) = |z_d - z_c| / (k + (z_d - z_c)^2)$, where $k$ should be
   $D_{\text{rot}}/D_{\text{lin}}$. The fitted $z_c$ must equal `compliance_point`.
   If it doesn't, the moment reference is at $z_{\text{ref,assumed}} + (z_c - z_{\text{point}})$.
2. **Free space.** $K = 0$; push by hand. Fit $v(t) = g\,f(t - L)$, picking $L$ by $R^2$.
   Compare with $g = 1/D$ and with forceMode's 0.93 mm/s per N and 90 ms.
3. **Tap.** Ramp toward a hard surface at 5–25 mm/s (0.3 s ramp, give up after 30 mm).
   Contact = 5 N; then the target sits `press`/K inside the surface (press 5 N) for 2 s. Report peak force, rise time,
   force reversals over 2 N in the 0.5 s after the peak, time below 1 N (lost contact),
   settling time and final force. The final force should be `press + deadband` = 6.5 N.
4. **Ramp (open loop).** Send trapezoid `speedL` profiles with no admittance. Compare
   95–160 Hz tracking error while ramping and while cruising. They should be equal
   (about 0.3 mm/s).
5. **Stall.** Command 5 mm/s, then go silent for 0.3 s: first without the watchdog
   (the arm keeps moving), then with it (C207A0 protective stop; the deceleration
   itself has not been logged yet).

**Safety summary.**
- Per cycle: the aborts in §3, step 0.
- Always clamped: $F$, the speed of $V$, and its acceleration.
- Stall: the watchdog.
- Impact: peak force is set by approach speed, not by the controller, since the rise
  takes 20–60 ms. Measured: peak ≈ 1.5 N + 1.08 N per mm/s at 5–17 mm/s, against
  2.0 N per mm/s for forceMode. Stay at 25 mm/s or less near contact; faster approaches
  haven't been measured with the current settings.

## 10. Log (one npz per run, one row per cycle)

The script logs **mixed** coordinates, as UR reports them:
- `t`; `pose` and `target` ($T_{se}$, $T_{se^\star}$ as $[p;\ \mathrm{rotvec}]$).
- `twist` (measured, $B(R_{se})V^b_e$) and `v_cmd` (sent).
- `v_point`: the state, at $c$, in base axes.
- `tcp_force` ($W$, raw) and `ft_raw` (`getFtRawWrench`).
- `wrench_point`: $F$ after §3 step 1, in base axes.
- `err`: the error in $c$ axes.
- `q`, `k_scale`, `selection`, `loop_dt`, `seg`/`rep`/`phase`, `ff_gain` ($g$, two halves).

Also save every setting (the whole config), the payload, CoG and TCP offset read at
startup, the git revision, the end reason, and the maximum loop gap. Earlier logs were
hard to compare because the settings weren't recorded.

## 11. Measured facts

Facts, not settings (they used to be `[measured]` in the toml). Logs are
`fdcc-*.npz` in the repo.

| quantity | value | source / note |
|---|---|---|
| forceMode free-space gain | 0.93 mm/s per N | UR forceMode baseline, centred on the flange |
| forceMode lag | 90 ms | |
| forceMode impact slope | 2.0 N per mm/s above the commanded force | |
| FDCC impact slope | 1.08 N per mm/s, intercept 1.5 N | `fdcc-tap-20260922-171421`, 5–17 mm/s real approach, M 15 kg |
| FDCC settling after impact | 0.05–0.25 s | ~50 ms at 10 mm/s; 120–250 ms at 20–25 mm/s commanded |
| 25 Hz phantom mass in getActualTCPForce | ~20 kg, 34 ms late | the arm's own acceleration; loop gain 1.55 at M 10 kg |
| speedL tracking delay | 26 ms | command → measured TCP velocity, at 25 Hz |
| apparent resistance while accelerating | 4.9 N at ~2 m/s² | low frequency; trips the fade at stick starts |
| joint 1 stiction breakaway | 12–22 Nm | why direct-torque mode failed; speedL's joint loops absorb it |
