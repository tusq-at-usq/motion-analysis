# pylint: disable=invalid-name, too-many-arguments, too-many-branches,
import numpy as np

def rts_smoother(xs, Ps, x_prs, P_prs, Fs, Qs):

    n = len(xs)
    if not all([n_i == n for n_i in [len(Ps), len(Fs), len(Qs), len(x_prs), len(P_prs)]]):
        print("ERROR: RTS smoother inputs are not same length")

    x_rts = [None]*n
    P_rts = [None]*n
    x_rts[-1] = xs[-1]
    P_rts[-1] = Ps[-1]
    for k in range(n-2, -1, -1):

        C = Ps[k]@(Fs[k+1].T)@np.linalg.inv(P_prs[k+1])
        x_rts[k] = xs[k] + C@(x_rts[k+1] - x_prs[k+1])
        P_rts[k] = Ps[k] + C@(P_rts[k+1] - P_prs[k+1])@(C.T)

    return x_rts, P_rts





    #  P_rts = np.array(Ps).copy()
    #  X_rts = np.array(Xs).copy()
    #  K = [None]*n
    #  Pp = [None]*n
        #  #  Pp[k] = (Fs[k]@P_rts[k])@Fs[k].T + Qs[k]
        #  #  K[k] = (P_rts[k]@Fs[k].T)@np.linalg.inv(Pp[k])
        #  #  X_rts[k] += K[k]@(X_rts[k+1] - (Fs[k]@X_rts[k]))
        #  #  X_rts[k][3:7] = X_rts[k][3:7]/np.linalg.norm(X_rts[k][3:7])
        #  #  P_rts[k] += (K[k]@(P_rts[k+1] - Pp[k]))@K[k].T

        #  Pp[k] = np.dot(np.dot(Fs[k+1], P_rts[k]), Fs[k].T) + Qs[k+1]
        #  K[k]  = np.dot(np.dot(P_rts[k], Fs[k+1].T), np.linalg.inv(Pp[k]))
        #  X_rts[k] += np.dot(K[k], X_rts[k] - np.dot(Fs[k+1], X_rts[k]))
        #  #  X_rts[k][3:7] = X_rts[k][3:7]/np.linalg.norm(X_rts[k][3:7])
        #  P_rts[k] += np.dot(np.dot(K[k], P_rts[k+1] - Pp[k]), K[k].T)
    return X_rts, P_rts
       
