"""Evaluation metrics for panoptic symbol spotting."""

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import MeanShift


def compute_semantic_accuracy(
    predictions: NDArray[np.int64],
    targets: NDArray[np.int64],
    mask: NDArray[np.bool_] | None = None,
) -> float:
    """Compute semantic segmentation accuracy.

    Args:
        predictions: Predicted class labels of shape (batch, N) or (N,).
        targets: Ground truth labels of shape (batch, N) or (N,).
        mask: Valid element mask.

    Returns:
        Accuracy as a float.
    """
    predictions = predictions.flatten()
    targets = targets.flatten()

    if mask is not None:
        mask = mask.flatten()
        valid = mask & (targets >= 0)
    else:
        valid = targets >= 0

    if not valid.any():
        return 0.0

    correct = (predictions[valid] == targets[valid]).sum()
    total = valid.sum()

    return float(correct / total)


def compute_iou(
    predictions: NDArray[np.int64],
    targets: NDArray[np.int64],
    num_classes: int,
    mask: NDArray[np.bool_] | None = None,
) -> dict[str, float]:
    """Compute Intersection over Union for each class.

    Args:
        predictions: Predicted class labels.
        targets: Ground truth labels.
        num_classes: Number of classes.
        mask: Valid element mask.

    Returns:
        Dictionary with per-class IoU and mean IoU.
    """
    predictions = predictions.flatten()
    targets = targets.flatten()

    if mask is not None:
        mask = mask.flatten()
        valid = mask & (targets >= 0)
        predictions = predictions[valid]
        targets = targets[valid]
    else:
        valid = targets >= 0
        predictions = predictions[valid]
        targets = targets[valid]

    ious = {}
    valid_ious = []

    for c in range(num_classes):
        pred_c = predictions == c
        target_c = targets == c

        intersection = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()

        if union > 0:
            iou = float(intersection / union)
            ious[f"iou_class_{c}"] = iou
            valid_ious.append(iou)

    ious["mean_iou"] = float(np.mean(valid_ious)) if valid_ious else 0.0

    return ious


def cluster_instances(
    embeddings: NDArray[np.float32],
    bandwidth: float = 0.5,
) -> NDArray[np.int64]:
    """Cluster instance embeddings using Mean Shift.

    Args:
        embeddings: Instance embeddings of shape (N, embed_dim).
        bandwidth: Bandwidth for Mean Shift clustering.

    Returns:
        Instance IDs of shape (N,).
    """
    if len(embeddings) == 0:
        return np.array([], dtype=np.int64)

    clustering = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    try:
        instance_ids = clustering.fit_predict(embeddings)
    except ValueError:
        # Fall back if clustering fails
        instance_ids = np.zeros(len(embeddings), dtype=np.int64)

    return instance_ids.astype(np.int64)


def compute_panoptic_quality(
    pred_semantic: NDArray[np.int64],
    pred_instances: NDArray[np.int64],
    gt_semantic: NDArray[np.int64],
    gt_instances: NDArray[np.int64],
    num_classes: int,
    iou_threshold: float = 0.5,
    mask: NDArray[np.bool_] | None = None,
) -> dict[str, float]:
    """Compute Panoptic Quality (PQ) metric.

    PQ = (TP / (TP + 0.5*FP + 0.5*FN)) * mean_IoU_of_matched

    Args:
        pred_semantic: Predicted semantic labels (N,).
        pred_instances: Predicted instance IDs (N,).
        gt_semantic: Ground truth semantic labels (N,).
        gt_instances: Ground truth instance IDs (N,).
        num_classes: Number of semantic classes.
        iou_threshold: IoU threshold for matching.
        mask: Valid element mask.

    Returns:
        Dictionary with PQ, SQ (segmentation quality), RQ (recognition quality).
    """
    if mask is not None:
        valid = mask & (gt_semantic >= 0)
        pred_semantic = pred_semantic[valid]
        pred_instances = pred_instances[valid]
        gt_semantic = gt_semantic[valid]
        gt_instances = gt_instances[valid]
    else:
        valid = gt_semantic >= 0
        pred_semantic = pred_semantic[valid]
        pred_instances = pred_instances[valid]
        gt_semantic = gt_semantic[valid]
        gt_instances = gt_instances[valid]

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou = 0.0

    for c in range(num_classes):
        # Get instances for this class
        pred_mask = pred_semantic == c
        gt_mask = gt_semantic == c

        if not gt_mask.any() and not pred_mask.any():
            continue

        pred_inst_ids = np.unique(pred_instances[pred_mask])
        gt_inst_ids = np.unique(gt_instances[gt_mask])

        # Remove invalid instance IDs
        pred_inst_ids = pred_inst_ids[pred_inst_ids >= 0]
        gt_inst_ids = gt_inst_ids[gt_inst_ids >= 0]

        matched_pred = set()
        matched_gt = set()

        # Match predictions to ground truth
        for gt_id in gt_inst_ids:
            gt_inst_mask = (gt_instances == gt_id) & gt_mask

            best_iou = 0.0
            best_pred_id = None

            for pred_id in pred_inst_ids:
                if pred_id in matched_pred:
                    continue

                pred_inst_mask = (pred_instances == pred_id) & pred_mask

                intersection = (gt_inst_mask & pred_inst_mask).sum()
                union = (gt_inst_mask | pred_inst_mask).sum()

                if union > 0:
                    iou = intersection / union
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_id = pred_id

            if best_iou >= iou_threshold and best_pred_id is not None:
                total_tp += 1
                total_iou += best_iou
                matched_pred.add(best_pred_id)
                matched_gt.add(gt_id)

        # Count false positives and false negatives
        total_fp += len(pred_inst_ids) - len(matched_pred)
        total_fn += len(gt_inst_ids) - len(matched_gt)

    # Compute PQ, SQ, RQ
    if total_tp == 0:
        pq = 0.0
        sq = 0.0
        rq = 0.0
    else:
        sq = total_iou / total_tp
        rq = total_tp / (total_tp + 0.5 * total_fp + 0.5 * total_fn)
        pq = sq * rq

    return {
        "pq": pq,
        "sq": sq,
        "rq": rq,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


class MetricsAccumulator:
    """Accumulates metrics over batches."""

    def __init__(self, num_classes: int):
        """Initialize accumulator.

        Args:
            num_classes: Number of semantic classes.
        """
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        """Reset accumulated metrics."""
        self.total_correct = 0
        self.total_samples = 0
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.pq_values: list[dict[str, float]] = []

    def update(
        self,
        pred_semantic: NDArray[np.int64],
        pred_instances: NDArray[np.int64],
        gt_semantic: NDArray[np.int64],
        gt_instances: NDArray[np.int64],
        mask: NDArray[np.bool_] | None = None,
    ) -> None:
        """Update metrics with a batch of predictions.

        Args:
            pred_semantic: Predicted semantic labels.
            pred_instances: Predicted instance IDs.
            gt_semantic: Ground truth semantic labels.
            gt_instances: Ground truth instance IDs.
            mask: Valid element mask.
        """
        # Flatten
        pred_semantic = pred_semantic.flatten()
        pred_instances = pred_instances.flatten()
        gt_semantic = gt_semantic.flatten()
        gt_instances = gt_instances.flatten()

        if mask is not None:
            mask = mask.flatten()
            valid = mask & (gt_semantic >= 0)
        else:
            valid = gt_semantic >= 0

        pred_semantic = pred_semantic[valid]
        pred_instances = pred_instances[valid]
        gt_semantic = gt_semantic[valid]
        gt_instances = gt_instances[valid]

        # Update accuracy
        self.total_correct += (pred_semantic == gt_semantic).sum()
        self.total_samples += len(pred_semantic)

        # Update confusion matrix
        for p, g in zip(pred_semantic, gt_semantic, strict=True):
            if 0 <= p < self.num_classes and 0 <= g < self.num_classes:
                self.confusion_matrix[g, p] += 1

        # Compute PQ for this batch
        pq = compute_panoptic_quality(
            pred_semantic,
            pred_instances,
            gt_semantic,
            gt_instances,
            self.num_classes,
        )
        self.pq_values.append(pq)

    def compute(self) -> dict[str, float]:
        """Compute final metrics.

        Returns:
            Dictionary with all metrics.
        """
        metrics: dict[str, float] = {}

        # Accuracy
        if self.total_samples > 0:
            metrics["accuracy"] = self.total_correct / self.total_samples
        else:
            metrics["accuracy"] = 0.0

        # Per-class IoU from confusion matrix
        ious = []
        for c in range(self.num_classes):
            tp = self.confusion_matrix[c, c]
            fn = self.confusion_matrix[c, :].sum() - tp
            fp = self.confusion_matrix[:, c].sum() - tp

            if tp + fn + fp > 0:
                iou = tp / (tp + fn + fp)
                ious.append(iou)
                metrics[f"iou_class_{c}"] = iou

        metrics["mean_iou"] = np.mean(ious) if ious else 0.0

        # Average PQ metrics
        if self.pq_values:
            metrics["pq"] = float(np.mean([v["pq"] for v in self.pq_values]))
            metrics["sq"] = float(np.mean([v["sq"] for v in self.pq_values]))
            metrics["rq"] = float(np.mean([v["rq"] for v in self.pq_values]))
        else:
            metrics["pq"] = 0.0
            metrics["sq"] = 0.0
            metrics["rq"] = 0.0

        return metrics
