import dataclasses
from abc import ABC
from typing import List

import numpy as np

from dudek.data.team_bas import SoccerVideo


@dataclasses.dataclass
class TeamBASScoredVideo:
    video: SoccerVideo
    scores: np.ndarray
    targets: np.ndarray


class TDeedMAPEvaluator(ABC):

    @staticmethod
    def compute_map(
        videos_data: List[TeamBASScoredVideo], delta_frames: int, num_classes: int
    ):
        """
        Compute mAP@delta_frames.

        Args:
            videos_data: List of VideoScoresTargets instances.
            delta_frames: Time tolerance in frames (delta).
            num_classes: Number of event classes.

        Returns:
            mAP: Mean Average Precision across all classes.
        """
        APs = []

        # Process each class separately
        for class_idx in range(num_classes):
            all_predictions = []  # List to store all predictions for this class
            all_ground_truths = {}  # Dict to store ground truths per video

            # Collect predictions and ground truths across all videos
            for vid_data in videos_data:
                video = vid_data.video
                vid_id = video.absolute_path
                predictions = vid_data.scores
                targets = vid_data.targets

                # Get predictions and ground truths for the current class
                class_preds = predictions[:, class_idx]
                class_targets = targets[:, class_idx]

                # Get prediction indices (frames) and confidence scores
                pred_indices = np.where(class_preds > 0)[0]
                pred_scores = class_preds[pred_indices]
                for idx, score in zip(pred_indices, pred_scores):
                    all_predictions.append(
                        {"video_id": vid_id, "frame_idx": idx, "score": score}
                    )

                # Get ground truth event indices (frames)
                gt_indices = np.where(class_targets == 1)[0]
                if vid_id not in all_ground_truths:
                    all_ground_truths[vid_id] = {
                        "gt_indices": gt_indices.tolist(),  # Ground truth frame indices
                        "matches": np.zeros(len(gt_indices), dtype=bool),  # Match flags
                    }
                else:
                    # In case of duplicate video IDs (unlikely), append ground truths
                    all_ground_truths[vid_id]["gt_indices"].extend(gt_indices.tolist())
                    all_ground_truths[vid_id]["matches"] = np.concatenate(
                        [
                            all_ground_truths[vid_id]["matches"],
                            np.zeros(len(gt_indices), dtype=bool),
                        ]
                    )

            # Sort predictions by confidence score in descending order
            all_predictions.sort(key=lambda x: x["score"], reverse=True)

            TP = np.zeros(len(all_predictions))
            FP = np.zeros(len(all_predictions))
            total_gt = sum(len(v["gt_indices"]) for v in all_ground_truths.values())

            # Process each prediction
            for idx, pred in enumerate(all_predictions):
                vid_id = pred["video_id"]
                frame_idx = pred["frame_idx"]
                gt_info = all_ground_truths.get(
                    vid_id, {"gt_indices": [], "matches": []}
                )
                gt_indices = gt_info["gt_indices"]
                matches = gt_info["matches"]

                # Find ground truths within delta_frames
                min_delta = float("inf")
                matched_gt_idx = -1
                for gt_idx, gt_frame in enumerate(gt_indices):
                    if not matches[gt_idx]:
                        delta = abs(frame_idx - gt_frame)
                        if delta <= delta_frames and delta < min_delta:
                            min_delta = delta
                            matched_gt_idx = gt_idx

                if matched_gt_idx >= 0:
                    # True Positive
                    TP[idx] = 1
                    matches[matched_gt_idx] = True
                    all_ground_truths[vid_id]["matches"] = matches
                else:
                    # False Positive
                    FP[idx] = 1

            # Compute cumulative True Positives and False Positives
            cum_TP = np.cumsum(TP)
            cum_FP = np.cumsum(FP)

            # Compute Precision and Recall
            precisions = cum_TP / (cum_TP + cum_FP + 1e-8)
            recalls = cum_TP / (total_gt + 1e-8)

            # Compute Average Precision (AP)
            AP = TDeedMAPEvaluator.compute_ap(recalls, precisions)
            APs.append(AP)

        # Compute Mean Average Precision (mAP)
        mAP = np.mean(APs)

        return mAP

    @staticmethod
    def compute_ap(recalls, precisions):
        """
        Compute the Average Precision (AP) using numerical integration.

        Args:
            recalls: Array of recall values.
            precisions: Array of precision values.

        Returns:
            AP: Average Precision.
        """
        # Append sentinel values at the end
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))

        # Compute the precision envelope
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # Calculate area under the curve
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        AP = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

        return AP
