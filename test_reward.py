import numpy as np

def get_force_reward(zf, min_f=1.0, large_f=5, max_f=10):
    r = np.zeros_like(zf); cond1 = min_f <= zf; cond2 = zf <= large_f; cond3 = large_f < zf; cond3_ = zf <= max_f; cond4 = zf > max_f 
    positive_mask = np.logical_and(cond1, cond2); negative_mask = np.logical_and(cond3_, cond3); negative_rewards = -( zf[negative_mask] - large_f )/(max_f - large_f)

    # assign rewards 
    if np.any(positive_mask):
        r[positive_mask] = 1.0 
    if np.any(negative_mask):
        r[negative_mask] = negative_rewards
    if np.any(cond4):
        r[cond4] = -1.0 
    return r


x  = np.array([-1.0, -0.5, 0.5, 1.5, 3.5, 5.5, 8.6, 10.0, 20])

print(get_force_reward(x))