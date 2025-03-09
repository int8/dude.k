from dudek.utils.video import load_bas_videos, load_action_spotting_videos

import click

cli = click.Group()


@cli.command()
@click.option("--dataset_path", type=str, required=True)
@click.option("--resolution", type=int, default=224)
@click.option("--stride", type=int, default=2)
@click.option("--frame_target_width", type=int, default=224)
@click.option("--frame_target_height", type=int, default=224)
@click.option("--grayscale", type=bool, default=False)
@click.option("--save_all", type=bool, default=False)
def extract_bas_frames(
    dataset_path: str,
    resolution: int = 224,
    stride: int = 2,
    frame_target_width: int = 224,
    frame_target_height: int = 224,
    grayscale: bool = False,
    save_all: bool = False,
):
    assert resolution in [224, 720]

    videos = load_bas_videos(dataset_path, resolution=resolution)
    for v in videos:
        if save_all:
            v.save_all_frames(
                target_width=frame_target_width,
                target_height=frame_target_height,
                stride=stride,
                grayscale=grayscale,
            )
        else:
            v.save_frames(
                target_width=frame_target_width,
                target_height=frame_target_height,
                stride=stride,
                grayscale=grayscale,
            )


@cli.command()
@click.option("--dataset_path", type=str, required=True)
@click.option("--resolution", type=int, default=224)
@click.option("--stride", type=int, default=2)
@click.option("--frame_target_width", type=int, default=224)
@click.option("--frame_target_height", type=int, default=224)
@click.option("--grayscale", type=bool, default=False)
@click.option("--radius_sec", type=int, default=8)
def extract_action_spotting_frames(
    dataset_path: str,
    resolution: int = 224,
    stride: int = 2,
    frame_target_width: int = 224,
    frame_target_height: int = 224,
    grayscale: bool = False,
    radius_sec: int = 8,
):
    assert resolution in [224, 720]

    videos = load_action_spotting_videos(dataset_path, resolution=resolution)
    for v in videos:
        v.save_frames(
            target_width=frame_target_width,
            target_height=frame_target_height,
            stride=stride,
            grayscale=grayscale,
            radius_around_annotations_in_sec=radius_sec,
        )


if __name__ == "__main__":
    cli()
