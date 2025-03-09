import dataclasses
import enum
import json
import os
import shutil
import time

from contextlib import nullcontext
from typing import List, Type, Optional


import numpy as np
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from dudek.data.team_bas import (
    SoccerVideo,
    Team,
    BASLabel,
    PredictedAnnotation,
)
from dudek.ml.data.tdeed import (
    TdeedVideoClip,
    TeamTDeedDataset,
    TeamTDeed2HeadsPrediction,
)
from dudek.ml.model.tdeed.eval.base import TeamBASScoredVideo, TDeedMAPEvaluator
from dudek.ml.model.tdeed.eval.legacy import mAPevaluateTest
from dudek.ml.model.tdeed.modules.tdeed import TDeedModule

from dudek.utils.common import (
    soft_non_maximum_suppression,
)


@dataclasses.dataclass
class _TeamBASScoredVideo(TeamBASScoredVideo):
    video: SoccerVideo
    scores: np.ndarray
    targets: np.ndarray
    annotations: List[PredictedAnnotation] = dataclasses.field(default=None, init=False)

    @classmethod
    def from_predictions(
        cls,
        video: SoccerVideo,
        predictions: List[TeamTDeed2HeadsPrediction],
        labels_enum: Type[enum.Enum],
        use_snms: bool = False,
        snms_params: dict = None,
    ):

        scores = TeamTDeed2HeadsPrediction.compute_team_scores_matrix(
            video, predictions, no_background=True, labels_enum=labels_enum
        )

        labels2int_map = {label: i + 1 for i, label in enumerate(labels_enum)}

        targets_left = video.compute_team_labels_matrix(
            team=Team.LEFT, no_background=True, labels2int_map=labels2int_map
        )

        targets_right = video.compute_team_labels_matrix(
            team=Team.RIGHT, no_background=True, labels2int_map=labels2int_map
        )

        targets = np.concatenate((targets_left, targets_right), axis=1)

        if use_snms:
            snms_params = snms_params or dict(class_window=12, threshold=0.01)
            scores = soft_non_maximum_suppression(scores, **snms_params)

        return cls(video=video, scores=scores, targets=targets)

    def annotate(self, labels_enum: Type[enum.Enum], use_true_fps: bool = False):

        predicted_annotations = []
        int2labels_map = TdeedVideoClip.get_int2label_map(
            labels_enum,
        )
        fps = self.video.actual_fps if use_true_fps else self.video.metadata_fps
        for i, x in enumerate(self.scores):
            confidence = np.max(x)
            label_idx = np.argmax(x)
            r = int2labels_map.get(label_idx + 1)
            if r is None:
                continue
            position = int(i / fps * 1000)
            label, team = r
            predicted_annotations.append(
                PredictedAnnotation(
                    label=label,
                    position=position,
                    team=team,
                    confidence=confidence,
                    game_time="unset",  # TODO
                    half=1 if ((position / 1000) / 60) > 45 else 0,  # let's say..
                )
            )

        self.annotations = predicted_annotations

    def add_flipped(self, flipped: "_TeamBASScoredVideo", labels_enum: Type[enum.Enum]):
        left = flipped.scores[:, 1 : (len(labels_enum) + 1)]
        right = flipped.scores[:, (len(labels_enum) + 1) : (2 * len(labels_enum) + 1)]
        flipped.scores[:, 1 : (len(labels_enum) + 1)] = right
        flipped.scores[:, (len(labels_enum) + 1) : (2 * len(labels_enum) + 1)] = left
        self.scores = (self.scores + flipped.scores) / 2


class BASTeamTDeedEvaluator(TDeedMAPEvaluator):
    def __init__(
        self,
        model: TDeedModule,
        dataset: TeamTDeedDataset,
        delta_frames_tolerance: int = 1,
    ):
        self.model = model
        self.dataset = dataset
        self.delta_frames_tolerance = delta_frames_tolerance

    def eval(
        self,
        batch_size: int = 32,
        use_snms: Optional[bool] = True,
        use_hflip: bool = False,
        snms_params: Optional[dict] = None,
    ):
        scored_videos = self.get_scored_videos(
            batch_size=batch_size,
            use_snms=use_snms,
            use_hflip=use_hflip,
            snms_params=snms_params,
        )
        self.create_solution_file(
            scored_videos=scored_videos,
            zip_output_file_name="/tmp/solution",
        )
        videos = [k for k, _ in self.dataset.group_by_videos().items()]
        games = [os.path.join(v.season, v.league, v.match) for v in videos]
        m = mAPevaluateTest(
            games,
            "/mnt/data4t/soccernet/sn_bas_2025/",
            "/tmp/solution.zip",
            event_team=True,
        )

        map_mine = self.compute_map(
            scored_videos, self.delta_frames_tolerance, 2 * len(BASLabel)
        )

        return m, map_mine

    def get_scored_videos(
        self,
        batch_size: int = 32,
        use_snms: Optional[bool] = False,
        use_hflip: bool = False,
        snms_params: Optional[dict] = None,
    ):
        video_dataset_map = self.dataset.group_by_videos()
        flipped_scores = None
        scored_videos: List[_TeamBASScoredVideo] = []
        for video, clips_dataset in video_dataset_map.items():
            video_predictions = []
            bas_predictions_flipped = []
            clips_loader = DataLoader(
                clips_dataset, batch_size=batch_size, collate_fn=lambda x: x
            )
            for batch_of_clips in tqdm(
                clips_loader,
                desc=f"Scoring video {video.absolute_path}",
            ):
                bas_predictions = self.predict(batch_of_clips)
                video_predictions += bas_predictions

            if use_hflip:
                clips_dataset.flip_proba = 1.0
                clips_loader = DataLoader(
                    clips_dataset, batch_size=batch_size, collate_fn=lambda x: x
                )
                for batch_of_clips in tqdm(
                    clips_loader,
                    desc=f"Scoring video {video.absolute_path}",
                ):
                    bas_predictions = self.predict(batch_of_clips)
                    bas_predictions_flipped += bas_predictions

                flipped_scores = _TeamBASScoredVideo.from_predictions(
                    video,
                    video_predictions,
                    use_snms=use_snms,
                    labels_enum=self.dataset.labels_enum,
                    snms_params=snms_params,
                )
            scored_video = _TeamBASScoredVideo.from_predictions(
                video,
                video_predictions,
                use_snms=use_snms,
                labels_enum=self.dataset.labels_enum,
                snms_params=snms_params,
            )

            if use_hflip:
                scored_video.scores = (scored_video.scores + flipped_scores.scores) / 2
            scored_videos.append(scored_video)
        return scored_videos

    def create_solution_file(
        self,
        scored_videos: List[_TeamBASScoredVideo],
        zip_output_file_name: str,
        json_filename: str = "results_spotting.json",
    ):

        tmp_root_dir = f"/tmp/.sn-{time.time()}"
        os.makedirs(tmp_root_dir, exist_ok=True)
        self.annotate(scored_videos)
        for scored_video in scored_videos:

            season = scored_video.video.season
            match = scored_video.video.match
            league = scored_video.video.league

            result_json = {
                "UrlLocal": os.path.join(season, league, match),
                "predictions": [],
            }

            os.makedirs(os.path.join(tmp_root_dir, season, league, match))
            results_json_file = os.path.join(
                tmp_root_dir, season, league, match, json_filename
            )

            for annotation in scored_video.annotations:
                result_json["predictions"].append(annotation.to_json())

            with open(results_json_file, "w") as f:
                json.dump(result_json, f)
        shutil.make_archive(zip_output_file_name, "zip", tmp_root_dir)
        shutil.rmtree(tmp_root_dir)

    def annotate(
        self,
        scored_videos: List[_TeamBASScoredVideo],
    ):
        for scored_video in scored_videos:
            scored_video.annotate(labels_enum=self.dataset.labels_enum)

        return scored_videos

    def predict(self, clips: List[TdeedVideoClip], use_amp=True, device: str = "cuda"):

        clips_tensor = torch.stack([c.clip_tensor for c in clips])

        if clips_tensor.device != device:
            clips_tensor = clips_tensor.to(device)
        clips_tensor = clips_tensor.float()
        self.model.eval()
        with torch.no_grad():
            with torch.amp.autocast("cuda") if use_amp else nullcontext():
                predictions, _ = self.model(clips_tensor, inference=True)
                return [
                    TeamTDeed2HeadsPrediction(
                        labels_prediction=predictions["im_feat"][i].squeeze(),
                        label_displacement_prediction=predictions["displ_feat"][
                            i
                        ].squeeze(),
                        clip=clip.origin_video_clip,
                    )
                    for i, clip in enumerate(clips)
                ]
