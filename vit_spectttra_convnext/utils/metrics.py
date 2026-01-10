import numpy as np
import pandas as pd
from sklearn import metrics

np.seterr(divide="ignore", invalid="ignore")


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class F1Meter:
    def __init__(self, average="binary"):
        self.average = average
        self.reset()

    def update(self, y_true, y_pred):
        self.y_true = np.concatenate([self.y_true, y_true])
        self.y_pred = np.concatenate([self.y_pred, y_pred])
        self.avg = metrics.f1_score(self.y_true, self.y_pred, average=self.average)

    def reset(self):
        self.y_true = np.array([])
        self.y_pred = np.array([])


class SensitivityMeter:
    def __init__(self, average="binary"):
        self.average = average
        self.reset()

    def update(self, y_true, y_pred):
        self.y_true = np.concatenate([self.y_true, y_true])
        self.y_pred = np.concatenate([self.y_pred, y_pred])
        self.avg = metrics.recall_score(
            self.y_true, self.y_pred, pos_label=1, average=self.average
        )

    def reset(self):
        self.y_true = np.array([])
        self.y_pred = np.array([])


class SpecificityMeter:
    def __init__(self, average="binary"):
        self.average = average
        self.reset()

    def update(self, y_true, y_pred):
        self.y_true = np.concatenate([self.y_true, y_true])
        self.y_pred = np.concatenate([self.y_pred, y_pred])
        self.avg = metrics.recall_score(
            self.y_true, self.y_pred, pos_label=0, average=self.average
        )

    def reset(self):
        self.y_true = np.array([])
        self.y_pred = np.array([])


class AccuracyMeter:
    def __init__(self):
        self.reset()

    def update(self, y_true, y_pred):
        self.y_true = np.concatenate([self.y_true, y_true])
        self.y_pred = np.concatenate([self.y_pred, y_pred])
        self.avg = metrics.balanced_accuracy_score(self.y_true, self.y_pred)

    def reset(self):
        self.y_true = np.array([])
        self.y_pred = np.array([])
def get_part_result(df: pd.DataFrame):
    """
    Partition test results by dataset 'bucket' and report ACC (balanced accuracy)
    and F1 for each bucket. Requires columns: y_true, y_pred.
    If 'bucket' is missing, we try to synthesize one from available columns, else
    everything is grouped under 'all'.
    """
    import numpy as np
    from sklearn import metrics

    needed = {"y_true", "y_pred"}
    if not needed.issubset(df.columns):
        raise ValueError(f"get_part_result needs columns {needed}")

    tdf = df.copy()

    # Ensure we have a 'bucket' column (your splitter already writes this).
    if "bucket" not in tdf.columns:
        if "algorithm" in tdf.columns and "source" in tdf.columns and "target" in tdf.columns:
            # Recreate bucket similar to splitter
            tdf["bucket"] = np.where(
                tdf["target"].astype(int) == 1,
                "ai::" + tdf["algorithm"].astype(str) + "::" + tdf["source"].astype(str),
                "human::" + tdf["source"].astype(str),
            )
        elif "source" in tdf.columns:
            tdf["bucket"] = tdf["source"].astype(str)
        else:
            tdf["bucket"] = "all"

    # Binarize predictions at 0.5
    y_bin = (tdf["y_pred"].astype(float).to_numpy() > 0.5).astype(int)
    tdf = tdf.assign(y_bin=y_bin, y_true=tdf["y_true"].astype(int))

    rows = []
    res_dict = {}

    for bucket, g in tdf.groupby("bucket"):
        y_true = g["y_true"].to_numpy()
        y_pred = g["y_bin"].to_numpy()

        acc = metrics.balanced_accuracy_score(y_true, y_pred)
        f1  = metrics.f1_score(y_true, y_pred, average="binary", zero_division=0)

        rows.append({
            "category": "bucket",
            "partition": str(bucket),
            "acc": float(acc),
            "f1": float(f1),
            "size": int(len(g)),
        })
        res_dict[f"bucket/{bucket}/acc"] = float(acc)
        res_dict[f"bucket/{bucket}/f1"]  = float(f1)

    part_result_df = pd.DataFrame(rows).sort_values(["partition"]).reset_index(drop=True)
    return part_result_df, res_dict
