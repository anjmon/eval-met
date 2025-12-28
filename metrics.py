import numpy as np


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return np.linalg.norm(y_true - y_pred) / np.sqrt(len(y_pred))


def accuracy(y_true, y_pred):
    if not np.issubdtype(y_true.dtype, np.integer):
        raise ValueError("Accuracy is only valid for categorical targets")
    return (y_true == y_pred).mean()


def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))


def mape(y_true, y_pred, eps=1e-7):
    return np.mean(np.abs((y_pred - y_true) / (y_true + eps))) * 100


METRICS = {
    "mse": mse,
    "rmse": rmse,
    "accuracy": accuracy,
    "mae": mae,
    "mape": mape,
}
