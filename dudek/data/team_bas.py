import dataclasses
import enum
import json
import os
from typing import Dict


import cv2
from typing import List, Optional, Type

from functools import cached_property

import numpy as np
import torch
import torchvision
from matplotlib.text import Annotation
from torchvision.transforms.functional import hflip
from tqdm import tqdm


from dudek.utils.frames import (
    save_frame,
    get_frame_numbers_around_centers,
)


class Team(enum.Enum):
    LEFT = "left"
    RIGHT = "right"
    NOT_APPLICABLE = "not applicable"

    def flip(self):
        if self == Team.LEFT:
            return Team.RIGHT
        elif self == Team.RIGHT:
            return Team.LEFT


class ActionLabel(enum.Enum):

    PENALTY = "Penalty"
    KICK_OFF = "Kick-off"
    GOAL = "Goal"
    SUBSTITUTION = "Substitution"
    OFFSIDE = "Offside"
    SHOTS_ON_TARGET = "Shots on target"
    SHOTS_OFF_TARGET = "Shots off target"
    CLEARANCE = "Clearance"
    BALL_OUT_OF_PLAY = "Ball out of play"
    THROW_IN = "Throw-in"
    FOUL = "Foul"
    INDIRECT_FREE_KICK = "Indirect free-kick"
    DIRECT_FREE_KICK = "Direct free-kick"
    CORNER = "Corner"
    YELLOW_CARD = "Yellow card"
    RED_CARD = "Red card"
    YELLOW_TO_RED_CARD = "Yellow->red card"

    def to_bas_label(self) -> Optional["BASLabel"]:

        if self == ActionLabel.GOAL:
            return BASLabel.GOAL
        elif self in [ActionLabel.SHOTS_OFF_TARGET, ActionLabel.SHOTS_ON_TARGET]:
            return BASLabel.SHOT
        elif self == ActionLabel.THROW_IN:
            return BASLabel.THROW_IN
        elif self in [ActionLabel.INDIRECT_FREE_KICK, ActionLabel.INDIRECT_FREE_KICK]:
            return BASLabel.FREE_KICK

        return None


class BASLabel(enum.Enum):
    PASS = "PASS"
    DRIVE = "DRIVE"
    HEADER = "HEADER"
    HIGH_PASS = "HIGH PASS"
    OUT = "OUT"
    CROSS = "CROSS"
    THROW_IN = "THROW IN"
    SHOT = "SHOT"
    BALL_PLAYER_BLOCK = "BALL PLAYER BLOCK"
    PLAYER_SUCCESSFUL_TACKLE = "PLAYER SUCCESSFUL TACKLE"
    FREE_KICK = "FREE KICK"
    GOAL = "GOAL"


@dataclasses.dataclass(frozen=True)
class Annotation:
    label: BASLabel | ActionLabel
    position: int
    game_time: str
    team: Team
    visibility: str

    def get_frame_nr(self, fps: float = 25) -> int:
        return int(self.position / 1000 * fps)

    @staticmethod
    def load_annotations(
        labels_path: str,
        half: Optional[int] = None,
        enum_class: Type[BASLabel | ActionLabel] = BASLabel,
        random_team_when_no_team: bool = False,
    ) -> List["Annotation"]:
        labels = None
        if os.path.exists(labels_path):
            labels = []
            with open(labels_path, "r") as f:
                labels_json = json.load(f)
                for label_json in labels_json["annotations"]:
                    if half is None or label_json["gameTime"][0] == str(half):
                        team = Team(label_json["team"])
                        if random_team_when_no_team and team == Team.NOT_APPLICABLE:
                            team = Team.RIGHT if np.random.rand() > 0.5 else Team.LEFT
                        labels.append(
                            Annotation(
                                label=enum_class(label_json["label"]),
                                position=int(label_json["position"]),
                                game_time=label_json["gameTime"],
                                team=team,
                                visibility=label_json["visibility"],
                            )
                        )
        return labels

    @staticmethod
    def load_bas_annotations(
        labels_path: Optional[str] = None,
    ) -> Optional[List["Annotation"]]:
        return Annotation.load_annotations(labels_path, enum_class=BASLabel)

    @staticmethod
    def load_action_annotations(
        labels_path: Optional[str] = None,
        half: Optional[int] = None,
        as_bas: bool = False,
        random_team_when_no_team: bool = False,
    ) -> Optional[List["Annotation"]]:
        annotations = Annotation.load_annotations(
            labels_path,
            half,
            enum_class=ActionLabel,
            random_team_when_no_team=random_team_when_no_team,
        )
        if as_bas:
            mapped_annotations = []
            for annotation in annotations or []:
                bas_label = annotation.label.to_bas_label()
                if bas_label:
                    mapped_annotation = Annotation(
                        label=bas_label,
                        position=annotation.position,
                        game_time=annotation.game_time,
                        team=annotation.team,
                        visibility=annotation.visibility,
                    )
                    mapped_annotations.append(mapped_annotation)
            return mapped_annotations

        return annotations

    def get_half(self):
        return int(self.game_time[0])


@dataclasses.dataclass(frozen=True)
class PredictedAnnotation:
    label: BASLabel | ActionLabel
    position: int
    team: Team
    game_time: str
    half: int
    confidence: float

    def to_json(self):
        return {
            "gameTime": self.game_time,
            "label": self.label.value,
            "position": int(self.position),
            "confidence": float(self.confidence),
            "half": self.half,
            "team": self.team.value,
        }


@dataclasses.dataclass(frozen=True)
class Frame:
    frame_path: str
    annotation: Optional[Annotation]

    @property
    def frame_filename(self):
        return self.frame_path.split("/")[-1]

    @property
    def original_video_frame_nr(self):
        return int(self.frame_filename.split(".")[0])

    def get_position(self, fps) -> int:
        return int((self.original_video_frame_nr / fps) * 1000)

    def to_bas(self):
        if (not self.annotation) or self.annotation.label in BASLabel:
            return self

        bas_label = self.annotation.label.to_bas_label()
        if bas_label:
            return Frame(
                frame_path=self.frame_path,
                annotation=Annotation(
                    label=bas_label,
                    position=self.annotation.position,
                    game_time=self.annotation.game_time,
                    team=self.annotation.team,
                    visibility=self.annotation.visibility,
                ),
            )
        else:
            return Frame(
                frame_path=self.frame_path,
                annotation=None,
            )


@dataclasses.dataclass
class SoccerVideo:
    season: str
    league: str
    match: str
    resolution: int
    absolute_path: str
    annotations: Optional[List[Annotation]]
    half: Optional[int] = dataclasses.field(default=None)
    labels_class: Type[BASLabel | ActionLabel] = dataclasses.field(default=None)

    cached_initial_frame: Optional[torch.Tensor] = dataclasses.field(default=None)

    def __hash__(self):
        return hash(self.absolute_path)

    @classmethod
    def bas_video_from_path(cls, video_dir_path: str, resolution: int) -> "SoccerVideo":

        if video_dir_path.endswith("/"):
            video_dir_path = video_dir_path[:-1]
            
        absolute_path = os.path.join(
            video_dir_path,
            f"{resolution}p.mp4",
        )
        season_name = video_dir_path.split("/")[-3]
        league_name = video_dir_path.split("/")[-2]
        match_label = video_dir_path.split("/")[-1]

        annotation_path = os.path.join(
            video_dir_path,
            "Labels-ball.json",
        )
        annotations = None
        if os.path.exists(annotation_path):
            annotations = Annotation.load_bas_annotations(annotation_path)

        return cls(
            resolution=resolution,
            absolute_path=absolute_path,
            annotations=annotations,
            season=season_name,
            league=league_name,
            match=match_label,
            half=None,
            labels_class=BASLabel,
        )

    @classmethod
    def action_spotting_video_from_path(
        cls,
        video_dir_path: str,
        resolution: int,
        half: Optional[int] = None,
        load_as_bas: bool = False,
        random_team_when_no_team: bool = False,
    ) -> "SoccerVideo" | List["SoccerVideo"]:

        season_name = video_dir_path.split("/")[-3]
        league_name = video_dir_path.split("/")[-2]
        match_label = video_dir_path.split("/")[-1]

        labels_v2_path = os.path.join(
            video_dir_path,
            "Labels-v2.json",
        )
        if half is None:
            half1 = cls.action_spotting_video_from_path(
                video_dir_path,
                resolution,
                half=1,
                load_as_bas=load_as_bas,
                random_team_when_no_team=random_team_when_no_team,
            )
            half2 = cls.action_spotting_video_from_path(
                video_dir_path,
                resolution,
                half=2,
                load_as_bas=load_as_bas,
                random_team_when_no_team=random_team_when_no_team,
            )
            return [half1, half2]

        assert half in [1, 2]
        absolute_path = os.path.join(
            video_dir_path,
            f"{half}_{resolution}p.mkv",
        )
        soccer_video = SoccerVideo(
            resolution=resolution,
            absolute_path=absolute_path,
            annotations=Annotation.load_action_annotations(
                labels_v2_path,
                half=half,
                as_bas=load_as_bas,
                random_team_when_no_team=random_team_when_no_team,
            ),
            season=season_name,
            league=league_name,
            match=match_label,
            half=half,
            labels_class=ActionLabel,
        )
        return soccer_video

    @property
    def id(self):
        return self.absolute_path

    def save_frames(
        self,
        stride: int = 2,
        target_height: int = 224,
        target_width: int = 224,
        radius_around_annotations_in_sec: Optional[int] = None,
        grayscale: bool = False,
    ):

        os.makedirs(self.frames_path, exist_ok=True)
        os.makedirs(self.grayscale_frames_path, exist_ok=True)
        if self.annotations is None:
            print(f"no annotations for {self.absolute_path}")
            forced_frame_numbers_to_keep = {}
            return
        else:
            forced_frame_numbers_to_keep = {a.get_frame_nr() for a in self.annotations}

        if radius_around_annotations_in_sec:
            forced_frame_numbers_to_keep = get_frame_numbers_around_centers(
                centers={a.get_frame_nr() for a in self.annotations},
                fps=self.metadata_fps,
                stride=stride,
                radius_in_sec=radius_around_annotations_in_sec,
            )

        for frame_nr, current_frame in tqdm(
            enumerate(self.play_video(grayscale=grayscale)),
            desc=f"saving frames to {self.frames_path}",
            total=self.metadata_n_frames,
        ):
            if radius_around_annotations_in_sec:
                if frame_nr not in forced_frame_numbers_to_keep:
                    continue

            if frame_nr % stride != 0 and frame_nr not in forced_frame_numbers_to_keep:
                continue
            save_frame(
                frame_nr,
                current_frame,
                self.frames_path if not grayscale else self.grayscale_frames_path,
                target_height,
                target_width,
            )

    def save_all_frames(
            self,
            stride: int = 2,
            target_height: int = 224,
            target_width: int = 224,
            grayscale: bool = False,
    ):

        os.makedirs(self.frames_path, exist_ok=True)
        os.makedirs(self.grayscale_frames_path, exist_ok=True)

        for frame_nr, current_frame in tqdm(
                enumerate(self.play_video(grayscale=grayscale)),
                desc=f"saving frames to {self.frames_path}",
                total=self.metadata_n_frames,
        ):

            if frame_nr % stride != 0:
                continue
            save_frame(
                frame_nr,
                current_frame,
                self.frames_path if not grayscale else self.grayscale_frames_path,
                target_height,
                target_width,
            )

    def get_clips(
        self, accepted_gap: int = 2, grayscale: bool = False
    ) -> List["VideoClip"]:
        clips = []
        current_clip = VideoClip.init_from_soccer_video(self)

        frames = self.grayscale_frames if grayscale else self.frames

        for frame in frames:
            if len(current_clip.frames) == 0:
                current_clip.frames.append(frame)
            else:
                if (
                    frame.original_video_frame_nr
                    - current_clip.frames[-1].original_video_frame_nr
                    > accepted_gap
                ):
                    clips.append(current_clip)
                    current_clip = VideoClip.init_from_soccer_video(self)
                current_clip.frames.append(frame)

        if len(current_clip.frames) > 0:
            clips.append(current_clip)

        return clips

    def play_video(self, grayscale: bool = False):
        vid = None
        try:
            vid = cv2.VideoCapture(self.absolute_path)
            while True:
                is_frame, frame = vid.read()
                if not is_frame:
                    break
                if grayscale:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                yield frame
        finally:
            if vid:
                vid.release()

    def compute_team_labels_matrix(
        self,
        labels2int_map: Dict[enum.Enum, int],
        team: Team,
        no_background: bool = False,
    ):
        labels_matrix = np.zeros(
            shape=(self.actual_n_frames, len(labels2int_map) + 1), dtype=np.float32
        )

        if self.annotations:
            for annotation in self.annotations:
                if annotation.team != team:
                    continue
                labels_matrix[
                    annotation.get_frame_nr(fps=self.actual_fps),
                    labels2int_map[annotation.label],
                ] = 1

        return labels_matrix if not no_background else labels_matrix[:, 1:]

    @cached_property
    def metadata_n_frames(self):
        vid = None
        try:
            vid = cv2.VideoCapture(self.absolute_path)
            return vid.get(cv2.CAP_PROP_FRAME_COUNT)
        finally:
            if vid:
                vid.release()

    @cached_property
    def frames_path(self):
        video_name = os.path.basename(self.absolute_path)
        dir_path = os.path.dirname(self.absolute_path)
        return os.path.join(dir_path, f".frames_{video_name}/")

    @cached_property
    def grayscale_frames_path(self):
        video_name = os.path.basename(self.absolute_path)
        dir_path = os.path.dirname(self.absolute_path)
        return os.path.join(dir_path, f".grayscale_frames_{video_name}/")

    @cached_property
    def actual_n_frames(self):
        n_frames = 0
        for _ in self.play_video():
            n_frames += 1
        return n_frames

    @cached_property
    def metadata_fps(self) -> float:
        vid = None
        try:
            vid = cv2.VideoCapture(self.absolute_path)
            return vid.get(cv2.CAP_PROP_FPS)
        finally:
            if vid:
                vid.release()

    @property
    def actual_fps(self):
        return self.actual_n_frames / (self.metadata_n_frames / self.metadata_fps)

    @cached_property
    def annotations_count(self):
        return len(self.annotations) if self.annotations is not None else 0

    @cached_property
    def _frame_nr_to_annotation_dict(self) -> Dict[int, Annotation]:
        frame_nr_to_annotation_dict = {}
        if self.annotations:
            for annotation in self.annotations:
                frame_nr = annotation.get_frame_nr(fps=self.metadata_fps)
                frame_nr_to_annotation_dict[frame_nr] = annotation
        return frame_nr_to_annotation_dict

    @cached_property
    def _frame_nr_to_position_dict(self) -> Dict[int, int]:
        frame_nr_to_position_dict = {}
        for frame in self.frames:
            position = frame.get_position(self.metadata_fps)
            frame_nr_to_position_dict[frame.original_video_frame_nr] = position
        return frame_nr_to_position_dict

    @property
    def video_type(self):
        if self.half is None:
            return "full"
        else:
            return f"half: {self.half}"

    @property
    def frames(self) -> List[Frame]:
        if os.path.exists(self.frames_path):
            frames = os.listdir(self.frames_path)
            frames.sort(key=lambda x: int(x.split(".")[0]))
            returned_frames = []
            for frame in frames:
                frame_path = os.path.join(self.frames_path, frame)
                annotations = self._frame_nr_to_annotation_dict.get(
                    int(frame.split(".")[0])
                )
                returned_frames.append(Frame(frame_path, annotations))
            return returned_frames
        else:
            raise FileNotFoundError(
                f"Frames at path {self.frames_path} do not exist - extract frames first"
            )

    @property
    def grayscale_frames(self) -> List[Frame]:
        if os.path.exists(self.grayscale_frames_path):
            frames = os.listdir(self.grayscale_frames_path)
            frames.sort(key=lambda x: int(x.split(".")[0]))
            returned_frames = []
            for frame in frames:
                frame_path = os.path.join(self.grayscale_frames_path, frame)
                annotations = self._frame_nr_to_annotation_dict.get(
                    int(frame.split(".")[0])
                )
                returned_frames.append(Frame(frame_path, annotations))
            return returned_frames
        else:
            raise FileNotFoundError(
                f"Frames at path {self.frames_path} do not exist - extract frames first"
            )

    def load_initial_frame(self, flip=False) -> torch.Tensor:

        if self.cached_initial_frame is None:
            self.cached_initial_frame = torchvision.io.read_image(
                self.frames[30].frame_path
            )

        return hflip(self.cached_initial_frame) if flip else self.cached_initial_frame

    def __repr__(self):
        return f"SoccerVideo(match='{self.match}', annotations_count={self.annotations_count}, video_type={self.video_type})"


@dataclasses.dataclass(frozen=True)
class VideoClip:
    frames: List[Frame]
    source_soccer_video: SoccerVideo
    labels_class: Type[BASLabel | ActionLabel]

    @classmethod
    def init_from_soccer_video(cls, soccer_video: SoccerVideo):
        return cls(
            frames=[],
            source_soccer_video=SoccerVideo(
                season=soccer_video.season,
                league=soccer_video.league,
                match=soccer_video.match,
                resolution=soccer_video.resolution,
                absolute_path=soccer_video.absolute_path,
                annotations=soccer_video.annotations,
                half=soccer_video.half,
            ),  # why not self? self may already contain cached properties - which are not needed here anyway
            labels_class=soccer_video.labels_class,
        )

    def split(
        self, clip_frames_count: int = 200, overlap: int = 50, to_bas: bool = False
    ):
        clips = []
        for i in range(0, len(self.frames), clip_frames_count - overlap):
            clip = VideoClip(
                self.frames[i : i + clip_frames_count],
                self.source_soccer_video,
                self.labels_class,
            )
            if len(clip.frames) != clip_frames_count:
                continue
            clips.append(clip) if not to_bas else clips.append(clip.to_bas())
        return clips

    def save_as_labelled_video(self, output_path: str, fps: float = 25):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        img = cv2.imread(self.frames[0].frame_path)
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            img.shape[:2][::-1],
        )

        for frame in self.frames:
            frame_img = cv2.imread(frame.frame_path)

            if frame.annotation:
                cv2.putText(
                    frame_img,
                    frame.annotation.label.name + f"[{frame.annotation.team.name}]",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                for _ in range(int(fps)):  # keep label for 1 second
                    out.write(frame_img)

            out.write(frame_img)
        cv2.destroyAllWindows()
        out.release()

    @cached_property
    def unique_annotations(self):
        unique_annotations = set()
        for frame in self.frames:
            if frame.annotation:
                unique_annotations.add(frame.annotation)
        return unique_annotations

    @cached_property
    def has_events(self) -> bool:
        for f in self.frames:
            if f.annotation:
                return True
        return False

    def get_half(self):
        for frame in self.frames:
            if frame.annotation:
                return frame.annotation.get_half()
        return None

    @cached_property
    def majority_team(self) -> Optional[Team]:
        # If there are no events in the clip or events come from both team left and right then returns None
        # if there are only events from one team then returns that team
        team = None
        for frame in self.frames:
            if frame.annotation:
                if team is None:
                    team = frame.annotation.team
                else:
                    if team != frame.annotation.team:
                        return None
        return team

    def to_bas(self):
        if self.labels_class == ActionLabel:
            return VideoClip(
                frames=[frame.to_bas() for frame in self.frames],
                source_soccer_video=self.source_soccer_video,
                labels_class=BASLabel,
            )
        else:
            return self


TEAM2INT = {Team.LEFT: 0, Team.RIGHT: 1}
INT2TEAM = {v: k for k, v in TEAM2INT.items()}
