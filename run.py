import numpy as np
import yaml

from metrics import METRICS


def load_array(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter=",")


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

    metric = cfg["metric"]
    if metric not in METRICS:
        raise ValueError(f"Unsupported metric: {metric}")

    score = METRICS[metric](y_true, y_pred)
    print({metric: float(score)})


if __name__ == "__main__":
    main()
