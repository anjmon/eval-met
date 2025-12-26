import numpy as np
import yaml


def load_array(path: str) -> np.array:
    return np.loadtxt(path, delimiter=",")


def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    y_true = load_array(cfg["y_true"])
    y_pred = load_array(cfg["y_pred"])

    if cfg["metric"] != "accuracy":
        raise ValueError(f"Unsupported metric: {cfg['metric']}")

    score = accuracy(y_true, y_pred)
    print({"accuracy": float(score)})


if __name__ == "__main__":
    main()
