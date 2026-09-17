"""
UR kinematics via pinocchio, in the frames the UR controller actually reports.

Validated against an independent nominal-DH implementation (agreement 5.9e-10 on
both FK and the Jacobian) and on hardware against `getActualTCPSpeed()`.
See test-impedance.py, which is the standalone probe this was lifted from.
"""
from scipy.spatial.transform import Rotation as R
import numpy as np

import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

# UR16e. A UR5e model on this arm is 76 mm of TCP error on average (131 mm worst
# case), which presents as a frame problem but is not one.
ROBOT_DESC = 'ur16e_description'


def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def pose_to_se3(p):
    return pin.SE3(R.from_rotvec(np.asarray(p, float)[3:]).as_matrix(),
                   np.asarray(p, float)[:3])


class URKin:
    """
    Everything returned here is in the `base` frame at the TCP.

    Frame choice is load-bearing and fails silently if wrong:
      * `base` vs `base_link` differ by exactly Rz(180 deg). `base_link` is the ROS
        REP-103 convention; `base` is what getActualTCPPose() is reported against.
        Using base_link negates commanded force in x and y -- the arm pushes the
        wrong way.
      * `flange` and `tool0` share an origin but not an orientation. `tool0` is the
        one matching the UR DH convention.
    """

    _models = {}

    def __init__(self, tcp_offset, desc=ROBOT_DESC, base='base', tool='tool0'):
        if desc not in URKin._models:
            URKin._models[desc] = load_robot_description(desc).model
        self.model = URKin._models[desc]
        self.data = self.model.createData()
        self.desc = desc
        self.f_base = self.model.getFrameId(base)
        self.f_tool = self.model.getFrameId(tool)
        self.tcp = pose_to_se3(tcp_offset)
        # Rated joint torques from the URDF rather than a hardcoded table.
        self.tau_rated = np.asarray(self.model.effortLimit, float)

    def fk(self, q):
        """base -> TCP as a UR pose vector [x, y, z, rx, ry, rz]."""
        pin.forwardKinematics(self.model, self.data, np.asarray(q, float))
        pin.updateFramePlacements(self.model, self.data)
        bMt = self.data.oMf[self.f_base].inverse() * self.data.oMf[self.f_tool] * self.tcp
        return np.r_[bMt.translation, R.from_matrix(bMt.rotation).as_rotvec()]

    def jacobian(self, q):
        """
        6x6 geometric Jacobian at the TCP in the `base` frame.
        Rows [vx vy vz wx wy wz], columns joints 1..6. ~16 us.
        """
        pin.computeJointJacobians(self.model, self.data, np.asarray(q, float))
        pin.updateFramePlacements(self.model, self.data)

        J = pin.getFrameJacobian(
            self.model, self.data, self.f_tool, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        ).copy()

        # Shift the reference point tool0 -> TCP: v_tcp = v_tool + w x r
        r = self.data.oMf[self.f_tool].rotation @ self.tcp.translation
        J[:3, :] -= skew(r) @ J[3:, :]

        # Rotate from the URDF root into the UR `base` frame.
        Rb = self.data.oMf[self.f_base].rotation
        return np.block([[Rb.T, np.zeros((3, 3))], [np.zeros((3, 3)), Rb.T]]) @ J

    @staticmethod
    def rescale_inertia(lam, model_diag, measured_diag):
        """
        Rescale sampled task-inertia matrices so their diagonal matches a
        MEASURED inertia, preserving the coupling structure.

        Chirp identification on this arm found the URDF-derived inertia wrong by
        1.5-4.2x on five axes and 0.48x on rz -- not a consistent factor, so no
        single scalar fixes it. Since the discrete damping bound is computed from
        Lambda^-1, an overestimate makes that bound too PERMISSIVE, which is why
        settings the model called safe still diverged. Correct it here:

            Lambda' = S Lambda S,   S = diag(sqrt(I_measured / I_model))

        which reproduces the measured diagonal exactly and scales the off-diagonal
        terms consistently.
        """
        s = np.sqrt(np.asarray(measured_diag, float) / np.asarray(model_diag, float))
        S = np.diag(s)
        return np.array([S @ L @ S for L in np.asarray(lam)])

    def sample_inertia(self, q0, spread=0.5, n=200, seed=0):
        """Full task-inertia matrices sampled around q0. Shape (n, 6, 6)."""
        rng = np.random.default_rng(seed)
        q0 = np.asarray(q0, float)
        return np.array([self.task_inertia(q0 + rng.uniform(-spread, spread, 6))
                         for _ in range(n)])

    def reference_inertia(self, q0, spread=0.5, n=200, seed=0):
        """
        Median per-axis task inertia around configuration q0, in the `base` frame
        at the TCP. Used to set the desired inertia and to derive damping.

        Compute this, never hardcode it. The TCP offset dominates the rotational
        terms -- at tool0 they are ~[0.08, 0.21, 0.03] kg m^2, but 154 mm out at
        the real TCP they are ~[0.44, 0.90, 0.15], a factor of 5-11. A stale value
        here silently underdamps every rotational axis and makes any inertia
        shaping wildly over-amplify.
        """
        lam = self.sample_inertia(q0, spread, n, seed)
        return np.median(np.diagonal(lam, axis1=1, axis2=2), axis=0)

    def task_inertia(self, q):
        """
        Task-space inertia (J M^-1 J^T)^-1 in the base frame. Not used in the
        control loop -- this is for choosing D offline via D = 2*zeta*sqrt(K*m).
        Measured median on this arm: 8.35 kg translational (x 12.2, y 7.5, z 7.6),
        0.085 kg m^2 rotational.
        """
        q = np.asarray(q, float)
        J = self.jacobian(q)
        pin.crba(self.model, self.data, q)
        M = np.triu(np.array(self.data.M))
        M = M + np.triu(M, 1).T
        return np.linalg.inv(J @ np.linalg.solve(M, J.T))
