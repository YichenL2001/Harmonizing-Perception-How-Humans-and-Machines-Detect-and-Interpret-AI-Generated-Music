import numpy as np
from sklearn import metrics


def calculate_stats(output, target):
    """Calculate statistics including mAP and AUC per class."""

    classes_num = target.shape[-1]
    stats = []

    acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(output, 1))

    for k in range(classes_num):
        avg_precision = metrics.average_precision_score(target[:, k], output[:, k], average=None)
        auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)

        precisions, recalls, _ = metrics.precision_recall_curve(target[:, k], output[:, k])
        fpr, tpr, thresholds = metrics.roc_curve(target[:, k], output[:, k])

        save_every_steps = 1000
        stats.append(
            {
                "precisions": precisions[0::save_every_steps],
                "recalls": recalls[0::save_every_steps],
                "AP": avg_precision,
                "fpr": fpr[0::save_every_steps],
                "fnr": 1.0 - tpr[0::save_every_steps],
                "auc": auc,
                "acc": acc,
            }
        )

    return stats
