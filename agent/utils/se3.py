"""
Twist / wrench actions in the chart of the local_delta pose actions.

A local_delta pose action is xi_i = log(T_0^-1 T_i)^v in scipy exp-coords order [w; v]
(StitchedSequenceDataset._pose_action_local_delta), relative to the chunk anchor T_0.
Twist and wrench actions live with it, at the anchor:

    twist   V^(0)_i = Ad_{T_0i} V^b_i            [w; v]    (se(3), so d xi/dt = J_l^-1(xi) V^(0))
    wrench  F^(0)_i = Ad_{T_0i}^-T F^b_i         [tau; f]  (se(3)*, <F^(0), V^(0)> = <F^b, V^b>)

with T_0i = exp(xi_i), and V^b_i / F^b_i the body twist / wrench of the TCP at T_i in fdcc
order [lin; ang] (fdcc.body_twist / fdcc.body_wrench, dataset fields twist_body /
wrench_body). The *_from_anchor functions invert them, back to body quantities at T_i --
what Env.step(des_twist=, des_wrench=) takes.

All functions take T_0i as (N, 4, 4) and values as (N, 6).
"""
import numpy as np

from robot.fdcc import adjoint


def _swap(x):
    """[a; b] <-> [b; a] on the last axis: fdcc's [lin; ang] <-> exp-coords [ang; lin]."""
    x = np.asarray(x, float)
    return np.concatenate([x[..., 3:], x[..., :3]], axis=-1)


def _ad(T):
    return np.stack([adjoint(t) for t in T])


def _inv(T):
    R, p = T[:, :3, :3], T[:, :3, 3]
    out = np.zeros_like(T)
    out[:, :3, :3] = np.transpose(R, (0, 2, 1))
    out[:, :3, 3] = -np.einsum('nji,nj->ni', R, p)
    out[:, 3, 3] = 1.0
    return out


def twist_to_anchor(T_0i, V_b):
    return _swap(np.einsum('nij,nj->ni', _ad(T_0i), V_b))


def twist_from_anchor(T_0i, a):
    return np.einsum('nij,nj->ni', _ad(_inv(T_0i)), _swap(a))


def wrench_to_anchor(T_0i, F_b):
    return _swap(np.einsum('nji,nj->ni', _ad(_inv(T_0i)), F_b))


def wrench_from_anchor(T_0i, a):
    return np.einsum('nji,nj->ni', _ad(T_0i), _swap(a))


TO_ANCHOR = {'twist': twist_to_anchor, 'wrench': wrench_to_anchor}
FROM_ANCHOR = {'twist': twist_from_anchor, 'wrench': wrench_from_anchor}
# feedforward action field -> dataset.h5 field of its body values (scripts/rawdata_to_dataset.py)
DATASET_FIELDS = {'twist': 'twist_body', 'wrench': 'wrench_body'}
