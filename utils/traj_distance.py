import math
import numpy as np
import time

# https://gist.github.com/MaxBareiss/ba2f9441d9455b56fbc9

# Euclidean distance.
def euc_dist(pt1,pt2):
    return math.sqrt((pt2[0]-pt1[0])*(pt2[0]-pt1[0])+(pt2[1]-pt1[1])*(pt2[1]-pt1[1]))

def _c(ca,i,j,P,Q):
    if ca[i,j] > -1:
        return ca[i,j]
    elif i == 0 and j == 0:
        ca[i,j] = euc_dist(P[0],Q[0])
    elif i > 0 and j == 0:
        ca[i,j] = max(_c(ca,i-1,0,P,Q),euc_dist(P[i],Q[0]))
    elif i == 0 and j > 0:
        ca[i,j] = max(_c(ca,0,j-1,P,Q),euc_dist(P[0],Q[j]))
    elif i > 0 and j > 0:
        ca[i,j] = max(min(_c(ca,i-1,j,P,Q),_c(ca,i-1,j-1,P,Q),_c(ca,i,j-1,P,Q)),euc_dist(P[i],Q[j]))
    else:
        ca[i,j] = float("inf")
    return ca[i,j]

def frechet_dist(P,Q):
    # P and Q are arrays of 2-element arrays (points)
    ca = np.ones((len(P),len(Q)))
    ca = np.multiply(ca,-1)
    return _c(ca,len(P)-1,len(Q)-1,P,Q)

# P=[[1,1], [2,1], [2,2]]
# Q=[[2,2], [0,1]]
# d = frechet_dist(P,Q)
# print(d)


def _get_linear_frechet(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    n_p = p.shape[0]
    n_q = q.shape[0]
    ca = np.zeros((n_p, n_q), dtype=np.float64)

    for i in range(n_p):
        for j in range(n_q):
            d = euc_dist(p[i], q[j])

            if i > 0 and j > 0:
                ca[i, j] = max(min(ca[i - 1, j],
                                   ca[i - 1, j - 1],
                                   ca[i, j - 1]), d)
            elif i > 0 and j == 0:
                ca[i, j] = max(ca[i - 1, 0], d)
            elif i == 0 and j > 0:
                ca[i, j] = max(ca[0, j - 1], d)
            elif i == 0 and j == 0:
                ca[i, j] = d
            else:
                ca[i, j] = np.infty
    return ca

# code ref: https://github.com/joaofig/discrete-frechet
def frechet_dist_linear(P, Q):
    p = np.array(P)
    q = np.array(Q)
    n_p = p.shape[0]
    n_q = q.shape[0]
    ca = _get_linear_frechet(p, q)
    return ca[-1,-1]


if __name__ == '__main__':
    p = [[80.0644976552576, 50.6552672944963],
                [71.4585771784186, 63.2156178820878],
                [19.9234400875866, 12.8415436018258]]

    q = [[5.88378887623549, 11.4293440245092],
                [84.2895035166293, 67.4984930083156],
                [90.9000392071903, 36.4088270813227],
                [34.2789062298834, 0.568102905526757],
                [43.9584670122713, 75.5553565453738],
                [24.4398877490312, 30.7297872845083],
                [35.2576361969113, 39.8860249202698],
                [62.438058713451, 44.4697478786111],
                [38.4228205773979, 66.4192265830934]]

    _t = time.time()
    for _ in range(10000):
        a = frechet_dist(p, q)
    print(time.time() - _t)

    _t = time.time()
    for _ in range(10000):
        a = frechet_dist_linear(p, q)
    print(time.time() - _t)