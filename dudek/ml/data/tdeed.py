import dataclasses
import enum
import random

from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from typing import List, Optional, Set, Dict, Type

import cv2
import numpy as np
import torch
import torchvision

from torch.utils.data import Dataset
from torchvision.transforms.v2.functional import hflip
from tqdm import tqdm

from dudek.config import DEFAULT_DEVICE
from dudek.data.team_bas import (
    VideoClip,
    BASLabel,
    SoccerVideo,
    Team,
)

from dudek.utils.common import linear_interpolate_row
from dudek.utils.frames import augment_with_camera_movement, crop_video


@dataclasses.dataclass
class TdeedVideoClip:

    origin_video_clip: VideoClip
    contains_event: bool
    labels_vector: torch.Tensor
    labels_displacement_vector: Optional[torch.Tensor]
    clip_tensor: torch.Tensor

    @classmethod
    def from_video_clip(
        cls,
        video_clip: VideoClip,
        labels_displacement: int = 0,
        flip_proba: float = 0.0,
        camera_movement_proba: float = 0.0,
        crop_proba: float = 0.0,
        crop_size: float = 0.9,
        labels_enum: Type[enum.Enum] = BASLabel,
        device=DEFAULT_DEVICE,
    ):
        labels2int_map = cls.get_labels2int_map(labels_enum)
        nr_of_classes = len(labels_enum) * 2 + 1

        num_frames = len(video_clip.frames)

        labels_vector = np.zeros(num_frames, dtype=np.int64)

        labels_displacement_vector = (
            np.zeros(num_frames, dtype=np.int64) if labels_displacement else None
        )
        flip = random.random() < flip_proba

        annotated_indices = []
        labels = []
        displacements = []

        for idx, frame in enumerate(video_clip.frames):
            if frame.annotation:

                team = frame.annotation.team
                if flip:
                    team = team.flip()

                label_value = labels2int_map[(frame.annotation.label, team)]

                start_disp = -labels_displacement
                end_disp = labels_displacement

                valid_disps = np.arange(
                    max(start_disp, -idx),
                    min(end_disp + 1, num_frames - idx),
                    dtype=np.int64,
                )

                displaced_indices = idx + valid_disps
                annotated_indices.extend(displaced_indices)
                labels.extend([label_value] * len(displaced_indices))

                if labels_displacement:
                    displacements.extend(valid_disps)

        if annotated_indices:
            annotated_indices = np.array(annotated_indices, dtype=np.int64)
            labels = np.array(labels, dtype=np.int64)
            labels_vector[annotated_indices] = labels

            if labels_displacement:
                displacements = np.array(displacements, dtype=np.int64)
                labels_displacement_vector[annotated_indices] = displacements

        frame_paths = [frame.frame_path for frame in video_clip.frames]

        def __load_image(path):

            img = torchvision.io.read_image(path)
            return hflip(img) if flip else img

        with ThreadPoolExecutor() as executor:
            imgs = list(executor.map(__load_image, frame_paths))

        # Stack images into a tensor
        clip_tensor = torch.stack(
            imgs,
            dim=0,
        )
        clip_tensor = clip_tensor.to(device)

        if random.random() < camera_movement_proba:
            clip_tensor = augment_with_camera_movement(clip_tensor)
        if random.random() < crop_proba:
            clip_tensor = crop_video(
                clip_tensor,
                crop_size_h=int(clip_tensor.shape[2] * crop_size),
                crop_size_w=int(clip_tensor.shape[3] * crop_size),
            )

        labels_vector = torch.Tensor(labels_vector).long()

        labels_vector = torch.nn.functional.one_hot(
            labels_vector, num_classes=nr_of_classes
        ).float()

        return cls(
            contains_event=video_clip.has_events,
            labels_vector=labels_vector.to(device),
            labels_displacement_vector=(
                torch.Tensor(labels_displacement_vector).to(device).float()
                if labels_displacement
                else None
            ),
            clip_tensor=clip_tensor.float(),
            origin_video_clip=video_clip,
        )

    @staticmethod
    def get_labels2int_map(labels_enum: Type[enum.Enum]):
        return {(label, Team.LEFT): i + 1 for i, label in enumerate(labels_enum)} | {
            (label, Team.RIGHT): i + 1 + len(labels_enum)
            for i, label in enumerate(labels_enum)
        }

    @staticmethod
    def get_int2label_map(labels_enum: Type[enum.Enum]):
        m = TdeedVideoClip.get_labels2int_map(labels_enum)
        return {v: k for k, v in m.items()}

    # noinspection PyTypeChecker
    def to_dict(self):
        return {
            "clip_tensor": self.clip_tensor,
            "labels_vector": self.labels_vector,
            "labels_displacement_vector": self.labels_displacement_vector,
        }

    def save_as_opencv_video(
        self, output_path: str, fps: float = 25, labels: Type[enum.Enum] = BASLabel
    ):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        first_frame = self.clip_tensor[0].cpu().permute(1, 2, 0).numpy()
        writer = cv2.VideoWriter(output_path, fourcc, fps, first_frame.shape[:2][::-1])
        for i, frame in enumerate(self.clip_tensor):
            frame = frame.detach().cpu().permute(1, 2, 0).numpy()
            compatible_frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)

            writer.write(compatible_frame)
        writer.release()


class TeamTDeedDataset(Dataset):
    def __init__(
        self,
        clips: List[VideoClip],
        labels_enum: Type[enum.Enum],
        displacement: int = 0,
        return_dict=True,
        flip_proba: float = 0.0,
        camera_move_proba: float = 0.0,
        crop_proba: float = 0.0,
        even_choice_proba: float = 0.0,
        enforced_epoch_size: int = None,
        evaluate: bool = False,
    ):
        self.clips = clips
        self.displacement = displacement
        self.return_dict = return_dict
        self.flip_proba = flip_proba
        self.camera_move_proba = camera_move_proba
        self.crop_proba = crop_proba
        self.labels_enum = labels_enum
        self.enforced_epoch_size = enforced_epoch_size
        self.eval = evaluate
        self.even_choice_proba = even_choice_proba

        if even_choice_proba > 0.0:
            self.clip_ids_by_label = {label: list() for label in labels_enum}
            for idx, clip in enumerate(self.clips):
                for annotation in clip.unique_annotations:
                    self.clip_ids_by_label[annotation.label] += [idx]

    def __len__(self):
        return (
            len(self.clips)
            if self.enforced_epoch_size is None
            else self.enforced_epoch_size
        )

    def __getitem__(self, idx):
        if self.enforced_epoch_size is not None:
            idx = random.choice(range(len(self.clips)))

        if self.even_choice_proba:
            proba = random.random()
            if proba < self.even_choice_proba:
                random_label = random.choice(list(self.clip_ids_by_label.keys()))
                label_idxs = self.clip_ids_by_label[random_label]
                idx = random.choice(label_idxs)
                bas_tdeed_video_clip = TdeedVideoClip.from_video_clip(
                    self.clips[idx],
                    self.displacement,
                    flip_proba=self.flip_proba * 2,
                    camera_movement_proba=self.camera_move_proba * 2,
                    crop_proba=self.crop_proba * 2,
                    labels_enum=self.labels_enum,
                )
                if self.return_dict:
                    return bas_tdeed_video_clip.to_dict()
                else:
                    return bas_tdeed_video_clip

        bas_tdeed_video_clip = TdeedVideoClip.from_video_clip(
            self.clips[idx],
            self.displacement,
            flip_proba=self.flip_proba,
            camera_movement_proba=self.camera_move_proba,
            crop_proba=self.crop_proba,
            labels_enum=self.labels_enum,
        )
        if self.return_dict:
            return bas_tdeed_video_clip.to_dict()
        else:
            return bas_tdeed_video_clip

    def get_unique_matches(self) -> Set[str]:
        unique_matches = set()
        for clip in self.clips:
            unique_matches.add(clip.source_soccer_video.match)
        return unique_matches

    def group_by_videos(
        self,
    ) -> Dict[SoccerVideo, "TeamTDeedDataset"]:
        videos = {}
        for clip in self.clips:
            if clip.source_soccer_video not in videos:
                videos[clip.source_soccer_video] = []
            videos[clip.source_soccer_video].append(clip)
        return {
            k: TeamTDeedDataset(
                v,
                labels_enum=self.labels_enum,
                displacement=self.displacement,
                return_dict=self.return_dict,
                flip_proba=self.flip_proba,
                camera_move_proba=self.camera_move_proba,
                crop_proba=self.crop_proba,
                even_choice_proba=self.even_choice_proba,
            )
            for k, v in videos.items()
        }

    def split_by_matches(
        self, counts: List[int], random_seed: int = 42
    ) -> List["TeamTDeedDataset"]:
        assert sum(counts) == len(
            self.get_unique_matches()
        ), f"Make sure your proportion sums up to {len(self.get_unique_matches())}"
        random.seed(random_seed)
        unique_matches = sorted(list(self.get_unique_matches()))
        random.shuffle(unique_matches)
        splits = []
        for count in counts:
            split_clips = [
                c
                for c in self.clips
                if c.source_soccer_video.match in unique_matches[:count]
            ]
            split_dataset = TeamTDeedDataset(
                split_clips,
                labels_enum=self.labels_enum,
                displacement=self.displacement,
                return_dict=self.return_dict,
                flip_proba=self.flip_proba,
                camera_move_proba=self.camera_move_proba,
                crop_proba=self.crop_proba,
                even_choice_proba=self.even_choice_proba,
            )
            splits.append(split_dataset)
            unique_matches = unique_matches[count:]
        assert sum([len(d.clips) for d in splits]) == len(self.clips)
        return splits

    def get_folds(self):
        videos = {}
        for clip in self.clips:
            if clip.source_soccer_video not in videos:
                videos[clip.source_soccer_video] = []
            videos[clip.source_soccer_video].append(clip)

        for k, v in videos.items():
            val = TeamTDeedDataset(
                v,
                labels_enum=self.labels_enum,
                displacement=self.displacement,
                return_dict=self.return_dict,
                flip_proba=self.flip_proba,
                camera_move_proba=self.camera_move_proba,
                crop_proba=self.crop_proba,
            )

            train = TeamTDeedDataset(
                [c for c in self.clips if c.source_soccer_video != k],
                labels_enum=self.labels_enum,
                displacement=self.displacement,
                return_dict=self.return_dict,
                flip_proba=self.flip_proba,
                camera_move_proba=self.camera_move_proba,
                crop_proba=self.crop_proba,
            )

            yield train, val


@dataclasses.dataclass
class TeamTDeed2HeadsPrediction:
    clip: VideoClip
    labels_prediction: torch.Tensor
    label_displacement_prediction: torch.Tensor

    @cached_property
    def displaced_label_predictions(self) -> np.ndarray:
        """
        Displaces the predicted events based on the predicted displacement
        """
        prediction_events = torch.softmax(self.labels_prediction, dim=1)

        def gaussian_weights(num_cols, center_col, sigma):
            cols = np.arange(num_cols)
            weights = np.exp(-((cols - center_col) ** 2) / (2 * sigma**2))
            return weights

        aux_pred = torch.zeros_like(prediction_events)

        for t, _ in enumerate(self.label_displacement_prediction):
            displacement = self.label_displacement_prediction[t].round().int().item()

            aux_pred[max(0, min(prediction_events.shape[0] - 1, t - displacement))] = (
                torch.maximum(
                    aux_pred[
                        max(0, min(prediction_events.shape[0] - 1, t - displacement))
                    ],
                    prediction_events[t],
                )
            )
        return aux_pred.detach().cpu().numpy()

    @cached_property
    def aligned_team_label_predictions(self):
        return self.align_with_original_video(self.displaced_label_predictions)

    def align_with_original_video(
        self, predictions_matrix: np.ndarray, interp: str = "linear"
    ):
        assert interp in ["copy", "linear"]
        start_frame = self.clip.frames[0].original_video_frame_nr
        end_frame = self.clip.frames[-1].original_video_frame_nr

        aligned_prediction_matrix = np.zeros(
            shape=(end_frame - start_frame + 1, predictions_matrix.shape[1])
        )
        aligned_prediction_matrix[:] = np.nan

        for i, frame in enumerate(self.clip.frames):
            events_predictions = predictions_matrix[i]
            if interp == "copy":
                aligned_prediction_matrix[
                    max(frame.original_video_frame_nr - start_frame - 1, 0)
                ] = events_predictions
            aligned_prediction_matrix[frame.original_video_frame_nr - start_frame] = (
                events_predictions
            )

        if interp == "linear":
            aligned_prediction_matrix = np.apply_along_axis(
                linear_interpolate_row, 0, aligned_prediction_matrix
            )

        return aligned_prediction_matrix

    @staticmethod
    def compute_team_scores_matrix(
        video: SoccerVideo,
        predictions: List["TeamTDeed2HeadsPrediction"],
        labels_enum: Type[enum.Enum],
        no_background: bool = False,
    ):

        scores_matrix = np.zeros(
            shape=(video.actual_n_frames, 2 * len(labels_enum) + 1), dtype=np.float32
        )

        support_matrix = np.zeros(
            shape=(video.actual_n_frames, 2 * len(labels_enum) + 1), dtype=np.float32
        )

        for prediction in tqdm(predictions, desc="constructing scores matrix"):
            predictions = prediction.aligned_team_label_predictions

            start_frame = prediction.clip.frames[0].original_video_frame_nr
            end_frame = prediction.clip.frames[-1].original_video_frame_nr

            scores_matrix[start_frame : (end_frame + 1)] = (
                scores_matrix[start_frame : (end_frame + 1)] + predictions
            )

            support_matrix[start_frame : (end_frame + 1)] = (
                support_matrix[start_frame : (end_frame + 1)] + 1
            )

        support_matrix[support_matrix == 0] = 1
        scores_matrix = scores_matrix / support_matrix

        return scores_matrix if not no_background else scores_matrix[:, 1:]
