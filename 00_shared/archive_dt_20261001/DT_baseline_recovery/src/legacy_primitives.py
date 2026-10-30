import numpy as np
def passive(x, mode_s, a):
    x = np.asarray(x)
    if mode_s:
        return np.maximum(0, a - x)
    return np.where(x <= a, a, 0)

def active(u, mode_s, v):
    u = np.asarray(u)
    if mode_s:
        return np.maximum(0, u - v)
    return np.where(u >= v, u, 0)

def D_apply(x, mode, a, b):
    p = passive(x, mode >> 1 & 1, a)
    A = active(x, mode & 1, b)
    return np.maximum(p, A).astype(np.int16)
