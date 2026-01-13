import numpy as np
import yaml

from metrics import METRICS, TASKS_METRICS_MAP


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
    task = cfg["task"]
    metrics = cfg["metrics"]

    for metric in metrics:
        if metric not in TASKS_METRICS_MAP[task]:
            print(
                f"{metric} is not a valid metric for {task}. \n Please use on of the following metrics - {TASKS_METRICS_MAP[task]}"
            )
            continue
        if metric not in METRICS:
            print(f"Unsupported metric: {metric}")
            continue

        print(f"Calculating {metric} for task {task}")

        score = METRICS[metric](y_true, y_pred)
        print({metric: float(score)})


if __name__ == "__main__":
    main()
