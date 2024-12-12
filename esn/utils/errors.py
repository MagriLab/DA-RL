import numpy as np


def L2(y, y_pred, axis=None):
    # equivalent to np.sqrt(np.sum((y-y_pred)**2, axis = axis))
    # frobenius norm is default
    return np.linalg.norm(y - y_pred, axis=axis)


def rel_L2(y, y_pred, axis=None):
    return L2(y, y_pred, axis=axis) / np.linalg.norm(y, axis=axis)


def mse(y, y_pred, axis=None):
    # mean squared error
    return np.mean((y - y_pred) ** 2, axis=axis)


def rmse(y, y_pred, axis=None):
    # root mean squared error
    return np.sqrt(mse(y, y_pred, axis))


def nrmse(y, y_pred, axis=None, normalize_by="rms"):
    # Calculate the normalized root mean squared error.
    if normalize_by == "rms":
        norm = np.sqrt(np.mean(y, axis=0) ** 2)
    elif normalize_by == "maxmin":
        norm = np.max(y, axis=0) - np.min(y, axis=0)
    elif normalize_by == "std":
        norm = np.std(y, axis=0) ** 2

    return np.mean(rmse(y, y_pred, axis=0) / norm, axis=axis)


def predictability_horizon(y, y_pred, t, LT):
    norm = np.std(y, axis=0) ** 2
    eps = np.sqrt(((y - y_pred) ** 2) / norm)
    tt = t / LT
    PH = np.zeros(y.shape[1])
    for i in range(y.shape[1]):
        PH_list = np.where(eps[:, i] > 0.5)[0]
        if len(PH_list) == 0:
            print("Predictability horizon longer than given time series.")
            PH[i] = tt[-1]
        else:
            PH[i] = tt[PH_list][0]
    return np.min(PH)
