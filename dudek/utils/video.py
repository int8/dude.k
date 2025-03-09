import os

from typing import List

from tqdm import tqdm

from dudek.data.team_bas import SoccerVideo, Annotation


def load_action_spotting_videos(
    directory_path: str,
    resolution: int,
    load_as_bas: bool = False,
    random_team_when_no_team: bool = False,
) -> List["SoccerVideo"]:
    soccer_videos = []
    with tqdm(desc=f"Loading videos from {directory_path}") as pbar:
        for league_name in os.listdir(directory_path):
            for season_name in os.listdir(os.path.join(directory_path, league_name)):
                for match_label in os.listdir(
                    os.path.join(directory_path, league_name, season_name)
                ):
                    half1, half2 = SoccerVideo.action_spotting_video_from_path(
                        os.path.join(
                            directory_path,
                            league_name,
                            season_name,
                            match_label,
                        ),
                        resolution,
                        load_as_bas=load_as_bas,
                        random_team_when_no_team=random_team_when_no_team,
                    )

                    soccer_videos.append(half1)
                    soccer_videos.append(half2)

                    pbar.update(1)
    return soccer_videos


def load_bas_videos(directory_path: str, resolution: int) -> List["SoccerVideo"]:
    videos = []
    for league_name in os.listdir(directory_path):
        for season_name in os.listdir(os.path.join(directory_path, league_name)):
            for match_label in os.listdir(
                os.path.join(directory_path, league_name, season_name)
            ):
                soccer_video = SoccerVideo.bas_video_from_path(
                    os.path.join(
                        directory_path,
                        league_name,
                        season_name,
                        match_label,
                    ),
                    resolution,
                )
                videos.append(soccer_video)
    return videos


import subprocess


def get_actual_video_length(video_file_path: str):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_file_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return float(result.stdout)
