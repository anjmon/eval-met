import numpy as np
import yaml


def load_array(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter=",")


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


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    y_true = load_array(cfg["y_true"])
    y_pred = load_array(cfg["y_pred"])

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    assert len(y_true) == len(y_pred), (
        "The ground truth and the predicted arrays must be of the same length!"
    )

    METRICS = {
        "mse": mse,
        "rmse": rmse,
        "accuracy": accuracy,
        "mae": mae,
        "mape": mape,
    }

    metric = cfg["metric"]
    if metric not in METRICS:
        raise ValueError(f"Unsupported metric: {metric}")

    score = METRICS[metric](y_true, y_pred)
    print({metric: float(score)})


if __name__ == "__main__":
    main()
