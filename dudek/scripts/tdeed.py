import torch
import wandb
from tqdm import tqdm

from dudek.config import (
    EXPERIMENTS_RANDOM_SEED,
    DEFAULT_DEVICE,
    TEST_SET_CHALLENGE_SEED,
)
from dudek.data.team_bas import ActionLabel, BASLabel
from dudek.ml.data.tdeed import TeamTDeedDataset
from dudek.ml.model.tdeed.eval.two_heads import BASTeamTDeedEvaluator
from dudek.ml.model.tdeed.modules.tdeed import TDeedModule


from dudek.ml.model.tdeed.training.two_heads import train as train_tdeed
from dudek.utils.video import load_action_spotting_videos, load_bas_videos

import click

cli = click.Group()


@cli.command()
@click.option("--dataset_path", type=str, required=True)
@click.option("--resolution", type=int, default=224)
@click.option("--clip_frames_count", type=int, default=70)
@click.option("--overlap", type=int, default=55)
@click.option("--displacement", type=int, default=4)
@click.option("--flip_proba", type=float, default=0.1)
@click.option("--camera_move_proba", type=float, default=0.1)
@click.option("--crop_proba", type=float, default=0.1)
@click.option("--even_choice_proba", type=float, default=0.0)
@click.option("--nr_epochs", type=int, default=40)
@click.option("--warm_up_epochs", type=int, default=1)
@click.option("--learning_rate", type=float, default=0.0006)
@click.option("--train_batch_size", type=int, default=8)
@click.option("--val_batch_size", type=int, default=8)
@click.option("--loss_foreground_weight", type=int, default=5)
@click.option("--eval_metric", type=str, default="loss")
@click.option("--start_eval_epoch_nr", type=int, default=0)
@click.option("--features_model_name", type=str, default="regnety_008")
@click.option("--temporal_shift_mode", type=str, default="gsf")
@click.option("--acc_grad_iter", type=int, default=1)
@click.option("--enforce_train_epoch_size", type=int, default=None)
@click.option("--enforce_val_epoch_size", type=int, default=None)
@click.option("--gaussian_blur_kernel_size", type=int, default=5)
@click.option("--tdeed_arch_n_layers", type=int, default=2)
@click.option("--tdeed_arch_sgp_ks", type=int, default=5)
@click.option("--tdeed_arch_sgp_k", type=int, default=4)
@click.option("--save_as", type=str, default="tdeed_pretrained.pt")
@click.option("--model_checkpoint_path", type=str, default=None)
@click.option("--experiment_name", type=str, default="tdeed_pretraining")
@click.option("--random_seed", type=int, default=EXPERIMENTS_RANDOM_SEED)
@click.option("--use_wandb", type=bool, default=False)
def pretrain(
    dataset_path: str,
    resolution: int = 224,
    clip_frames_count: int = 80,
    overlap: int = 60,
    displacement: int = 4,
    flip_proba: float = 0.1,
    camera_move_proba: float = 0.1,
    crop_proba: float = 0.1,
    even_choice_proba: float = 0.0,
    nr_epochs: int = 25,
    warm_up_epochs: int = 1,
    learning_rate: float = 0.0006,
    train_batch_size: int = 8,
    val_batch_size: int = 8,
    loss_foreground_weight: int = 5,  # how to weight event class in loss function <= background weight is 1,
    eval_metric="loss",
    start_eval_epoch_nr: int = 0,
    features_model_name: str = "regnety_008",
    temporal_shift_mode: str = "gsf",
    acc_grad_iter: int = 1,
    enforce_train_epoch_size: int = None,
    enforce_val_epoch_size: int = None,
    gaussian_blur_kernel_size: int = 5,
    tdeed_arch_n_layers: int = 2,
    tdeed_arch_sgp_ks: int = 5,
    tdeed_arch_sgp_k: int = 4,
    save_as: str = "tdeed_pretrained.pt",
    model_checkpoint_path: str = None,
    experiment_name: str = "tdeed_pretraining",
    random_seed: int = EXPERIMENTS_RANDOM_SEED,
    use_wandb: bool = True
):
    assert resolution in [224, 720]

    if use_wandb:
        wandb.init(project=experiment_name, sync_tensorboard=True)

    videos = load_action_spotting_videos(
        dataset_path, resolution=resolution, random_team_when_no_team=True
    )
    all_clips = []
    for v in tqdm(videos, desc="Loading clips ...", total=len(videos)):
        clips = v.get_clips(
            accepted_gap=2,
        )
        for clip in clips:
            all_clips += clip.split(
                clip_frames_count=clip_frames_count, overlap=overlap
            )

    all_dataset = TeamTDeedDataset(
        all_clips,
        labels_enum=ActionLabel,
        displacement=displacement,
        return_dict=True,
        flip_proba=flip_proba,
        camera_move_proba=camera_move_proba,
        crop_proba=crop_proba,
        even_choice_proba=even_choice_proba,
    )

    train_dataset, val_dataset = all_dataset.split_by_matches(
        counts=[460, 40], random_seed=random_seed
    )

    if enforce_train_epoch_size is not None:
        train_dataset.enforced_epoch_size = enforce_train_epoch_size

    if eval_metric == "map":
        val_dataset.return_dict = False
    val_dataset.flip_proba = 0.0
    val_dataset.camera_move_proba = 0.0
    val_dataset.crop_proba = 0.0
    val_dataset.even_choice_proba = 0.0

    if enforce_val_epoch_size is not None:
        val_dataset.enforced_epoch_size = enforce_val_epoch_size

    tdeed_model = TDeedModule(
        clip_len=clip_frames_count,
        n_layers=tdeed_arch_n_layers,
        sgp_ks=tdeed_arch_sgp_ks,
        sgp_k=tdeed_arch_sgp_k,
        num_classes=len(ActionLabel) * 2,
        features_model_name=features_model_name,
        temporal_shift_mode=temporal_shift_mode,
        gaussian_blur_ks=gaussian_blur_kernel_size,
    )

    if model_checkpoint_path:
        tdeed_model.load_all(model_checkpoint_path)

    train_tdeed(
        experiment_name=experiment_name,
        model=tdeed_model,
        labels_enum=ActionLabel,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        eval_metric=eval_metric,
        nr_epochs=nr_epochs,
        start_eval_epoch_nr=start_eval_epoch_nr,
        foreground_weight=loss_foreground_weight,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        device=DEFAULT_DEVICE,
        acc_grad_iter=acc_grad_iter,
        warm_up_epochs=warm_up_epochs,
        save_as=save_as,
        lr=learning_rate,
    )


@cli.command()
@click.option("--dataset_path", type=str, required=True)
@click.option("--resolution", type=int, default=224)
@click.option("--clip_frames_count", type=int, default=70)
@click.option("--overlap", type=int, default=55)
@click.option("--displacement", type=int, default=4)
@click.option("--flip_proba", type=float, default=0.1)
@click.option("--camera_move_proba", type=float, default=0.1)
@click.option("--crop_proba", type=float, default=0.1)
@click.option("--even_choice_proba", type=float, default=0.0)
@click.option("--nr_epochs", type=int, default=40)
@click.option("--warm_up_epochs", type=int, default=1)
@click.option("--learning_rate", type=float, default=0.0006)
@click.option("--train_batch_size", type=int, default=8)
@click.option("--val_batch_size", type=int, default=8)
@click.option("--eval_metric", type=str, default="map")
@click.option("--start_eval_epoch_nr", type=int, default=0)
@click.option("--loss_foreground_weight", type=int, default=5)
@click.option("--features_model_name", type=str, default="regnety_008")
@click.option("--temporal_shift_mode", type=str, default="gsf")
@click.option("--acc_grad_iter", type=int, default=1)
@click.option("--enforce_train_epoch_size", type=int, default=None)
@click.option("--enforce_val_epoch_size", type=int, default=None)
@click.option("--gaussian_blur_kernel_size", type=int, default=5)
@click.option("--tdeed_arch_n_layers", type=int, default=2)
@click.option("--tdeed_arch_sgp_ks", type=int, default=5)
@click.option("--tdeed_arch_sgp_k", type=int, default=4)
@click.option("--model_checkpoint_path", type=str, default=None)
@click.option("--save_as", type=str, default="tdeed_best.pt")
@click.option("--experiment_name", type=str, default="tdeed_training")
@click.option("--random_seed", type=int, default=TEST_SET_CHALLENGE_SEED)
@click.option("--use_wandb", type=bool, default=False)
def train(
    dataset_path: str,
    resolution: int = 224,
    clip_frames_count: int = 80,
    overlap: int = 60,
    displacement: int = 4,
    flip_proba: float = 0.2,
    camera_move_proba: float = 0.2,
    crop_proba: float = 0.2,
    even_choice_proba: float = 0.0,
    nr_epochs: int = 25,
    warm_up_epochs: int = 1,
    learning_rate: float = 0.0006,
    train_batch_size: int = 8,
    val_batch_size: int = 8,
    eval_metric="map",
    start_eval_epoch_nr: int = 0,
    loss_foreground_weight: int = 5,  # how to weight event class in loss function <= background weight is 1,
    features_model_name: str = "regnety_008",
    temporal_shift_mode: str = "gsf",
    acc_grad_iter: int = 1,
    enforce_train_epoch_size: int = None,
    enforce_val_epoch_size: int = None,
    gaussian_blur_kernel_size: int = 5,
    tdeed_arch_n_layers: int = 2,
    tdeed_arch_sgp_ks: int = 5,
    tdeed_arch_sgp_k: int = 4,
    model_checkpoint_path: str = None,
    save_as: str = "tdeed_best.pt",
    experiment_name: str = "tdeed_training",
    random_seed: int = TEST_SET_CHALLENGE_SEED,
    use_wandb: bool = False,
):
    assert resolution in [224, 720]
    if use_wandb:
        wandb.init(project=experiment_name, sync_tensorboard=True)


    videos = load_bas_videos(dataset_path, resolution=resolution)
    all_clips = []
    for v in tqdm(videos, desc="Loading clips ..."):
        clips = v.get_clips(
            accepted_gap=2,
        )
        for clip in clips:
            short_clips = clip.split(
                clip_frames_count=clip_frames_count, overlap=overlap
            )
            all_clips += short_clips

    all_dataset = TeamTDeedDataset(
        all_clips,
        labels_enum=BASLabel,
        displacement=displacement,
        return_dict=True,
        flip_proba=flip_proba,
        camera_move_proba=camera_move_proba,
        crop_proba=crop_proba,
        even_choice_proba=even_choice_proba,
    )

    train_dataset, val_dataset, test_dataset = all_dataset.split_by_matches(
        counts=[4, 1, 2], random_seed=random_seed
    )

    if enforce_train_epoch_size is not None:
        train_dataset.enforced_epoch_size = enforce_train_epoch_size

    if eval_metric == "map":
        val_dataset.return_dict = False

    val_dataset.flip_proba = 0.0
    val_dataset.camera_move_proba = 0.0
    val_dataset.crop_proba = 0.0
    val_dataset.even_choice_proba = 0.0

    if enforce_val_epoch_size is not None:
        val_dataset.enforced_epoch_size = enforce_val_epoch_size

    tdeed_model = TDeedModule(
        clip_len=clip_frames_count,
        n_layers=tdeed_arch_n_layers,
        sgp_ks=tdeed_arch_sgp_ks,
        sgp_k=tdeed_arch_sgp_k,
        num_classes=len(BASLabel) * 2,
        features_model_name=features_model_name,
        temporal_shift_mode=temporal_shift_mode,
        gaussian_blur_ks=gaussian_blur_kernel_size,
    )
    if model_checkpoint_path:
        tdeed_model.load_backbone(model_weight_path=model_checkpoint_path)

    train_tdeed(
        experiment_name=experiment_name,
        model=tdeed_model,
        labels_enum=BASLabel,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        eval_metric=eval_metric,
        nr_epochs=nr_epochs,
        start_eval_epoch_nr=start_eval_epoch_nr,
        foreground_weight=loss_foreground_weight,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        device=DEFAULT_DEVICE,
        acc_grad_iter=acc_grad_iter,
        save_as=save_as,
        lr=learning_rate,
        loss_weights=[1.5, 1],
        warm_up_epochs=warm_up_epochs,
    )


@cli.command()
@click.option("--dataset_path", type=str, required=True)
@click.option("--resolution", type=int, default=224)
@click.option("--clip_frames_count", type=int, default=80)
@click.option("--overlap", type=int, default=68)
@click.option("--displacement", type=int, default=4)
@click.option("--flip_proba", type=float, default=0.1)
@click.option("--camera_move_proba", type=float, default=0.1)
@click.option("--crop_proba", type=float, default=0.1)
@click.option("--even_choice_proba", type=float, default=0.0)
@click.option("--nr_epochs", type=int, default=40)
@click.option("--warm_up_epochs", type=int, default=1)
@click.option("--learning_rate", type=float, default=0.0008)
@click.option("--train_batch_size", type=int, default=8)
@click.option("--loss_foreground_weight", type=int, default=5)
@click.option("--features_model_name", type=str, default="regnety_008")
@click.option("--temporal_shift_mode", type=str, default="gsf")
@click.option("--acc_grad_iter", type=int, default=1)
@click.option("--enforce_train_epoch_size", type=int, default=None)
@click.option("--gaussian_blur_kernel_size", type=int, default=5)
@click.option("--tdeed_arch_n_layers", type=int, default=2)
@click.option("--tdeed_arch_sgp_ks", type=int, default=5)
@click.option("--tdeed_arch_sgp_k", type=int, default=4)
@click.option("--save_as", type=str, default="tdeed_challenge.pt")
@click.option("--model_checkpoint_path", type=str, default="tdeed_pretrained.pt")
@click.option("--experiment_name", type=str, default=None)
@click.option("--use_wandb", type=bool, default=False)
def train_challenge(
    dataset_path: str,
    resolution: int = 224,
    clip_frames_count: int = 80,
    overlap: int = 68,
    displacement: int = 4,
    flip_proba: float = 0.1,
    camera_move_proba: float = 0.1,
    crop_proba: float = 0.1,
    even_choice_proba: float = 0.0,
    nr_epochs: int = 40,
    warm_up_epochs: int = 1,
    learning_rate: float = 0.0008,
    train_batch_size: int = 8,
    loss_foreground_weight: int = 5,  # how to weight event class in loss function <= background weight is 1,
    features_model_name: str = "regnety_008",
    temporal_shift_mode: str = "gsf",
    acc_grad_iter: int = 1,
    enforce_train_epoch_size: int = None,
    gaussian_blur_kernel_size: int = 5,
    tdeed_arch_n_layers: int = 2,
    tdeed_arch_sgp_ks: int = 5,
    tdeed_arch_sgp_k: int = 4,
    save_as: str = "tdeed_challenge.pt",
    model_checkpoint_path: str = None,
    experiment_name: str = "tdeed_training_challenge",
    use_wandb: bool = False
):
    assert resolution in [224, 720]
    if use_wandb:
        wandb.init(project=experiment_name, sync_tensorboard=True)

    videos = load_bas_videos(dataset_path, resolution=resolution)
    all_clips = []
    for v in tqdm(videos, desc="Loading clips ..."):
        clips = v.get_clips(
            accepted_gap=2,
        )
        for clip in clips:
            short_clips = clip.split(
                clip_frames_count=clip_frames_count, overlap=overlap
            )
            all_clips += short_clips

    train_dataset = TeamTDeedDataset(
        all_clips,
        labels_enum=BASLabel,
        displacement=displacement,
        return_dict=True,
        flip_proba=flip_proba,
        camera_move_proba=camera_move_proba,
        crop_proba=crop_proba,
        even_choice_proba=even_choice_proba,
    )
    if enforce_train_epoch_size is not None:
        train_dataset.enforced_epoch_size = enforce_train_epoch_size

    tdeed_model = TDeedModule(
        clip_len=clip_frames_count,
        n_layers=tdeed_arch_n_layers,
        sgp_ks=tdeed_arch_sgp_ks,
        sgp_k=tdeed_arch_sgp_k,
        num_classes=len(BASLabel) * 2,
        features_model_name=features_model_name,
        temporal_shift_mode=temporal_shift_mode,
        gaussian_blur_ks=gaussian_blur_kernel_size,
    )

    if model_checkpoint_path:
        tdeed_model.load_backbone(model_weight_path=model_checkpoint_path)

    model = train_tdeed(
        experiment_name=experiment_name,
        model=tdeed_model,
        labels_enum=BASLabel,
        train_dataset=train_dataset,
        nr_epochs=nr_epochs,
        foreground_weight=loss_foreground_weight,
        train_batch_size=train_batch_size,
        device=DEFAULT_DEVICE,
        acc_grad_iter=acc_grad_iter,
        lr=learning_rate,
        loss_weights=[1.5, 1],
        warm_up_epochs=warm_up_epochs,
        eval_metric="loss",
        val_dataset=None,
        start_eval_epoch_nr=0,
    )

    torch.save(
        model.state_dict(),
        save_as,
    )



@cli.command()
@click.option("--dataset_path", type=str, required=True)
@click.option("--resolution", type=int, default=224)
@click.option("--clip_frames_count", type=int, default=80)
@click.option("--overlap", type=int, default=68)
@click.option("--displacement", type=int, default=4)
@click.option("--val_batch_size", type=int, default=8)
@click.option("--features_model_name", type=str, default="regnety_008")
@click.option("--temporal_shift_mode", type=str, default="gsf")
@click.option("--t_deed_arch_n_layers", type=int, default=2)
@click.option("--t_deed_arch_sgp_ks", type=int, default=5)
@click.option("--t_deed_arch_sgp_k", type=int, default=4)
@click.option("--model_checkpoint_path", type=str, default="tdeed_challenge.pt")
@click.option("--solution_archive_file_base_name", type=str, default="solution")
def create_solution(
    dataset_path: str,
    resolution: int = 224,
    clip_frames_count: int = 80,
    overlap: int = 68,
    displacement: int = 4,
    val_batch_size: int = 8,
    features_model_name: str = "regnety_008",
    temporal_shift_mode: str = "gsf",
    t_deed_arch_n_layers: int = 2,
    t_deed_arch_sgp_ks: int = 5,
    t_deed_arch_sgp_k: int = 4,
    model_checkpoint_path: str = "tdeed_challenge.pt",
    solution_archive_file_base_name: str = "solution",
):
    assert resolution in [224, 720]
    videos = load_bas_videos(dataset_path, resolution=resolution)
    all_clips = []
    for v in tqdm(videos, desc="Loading clips ..."):
        clips = v.get_clips(
            accepted_gap=2,
        )
        for clip in clips:
            all_clips += clip.split(
                clip_frames_count=clip_frames_count, overlap=overlap
            )

    challenge_dataset = TeamTDeedDataset(
        all_clips,
        labels_enum=BASLabel,
        displacement=displacement,
        return_dict=False,
        flip_proba=0.0,
        camera_move_proba=0.0,
        crop_proba=0.0,
        even_choice_proba=0.0,
    )

    assert challenge_dataset.flip_proba == 0.0
    assert challenge_dataset.camera_move_proba == 0.0
    assert challenge_dataset.crop_proba == 0.0
    assert challenge_dataset.even_choice_proba == 0.0

    tdeed_model = TDeedModule(
        clip_len=clip_frames_count,
        n_layers=t_deed_arch_n_layers,
        sgp_ks=t_deed_arch_sgp_ks,
        sgp_k=t_deed_arch_sgp_k,
        num_classes=len(BASLabel) * 2,
        features_model_name=features_model_name,
        temporal_shift_mode=temporal_shift_mode,
    )
    tdeed_model.load_all(model_weight_path=model_checkpoint_path)
    tdeed_model = tdeed_model.to(DEFAULT_DEVICE)
    tdeed_model.eval()
    evaluator = BASTeamTDeedEvaluator(
        model=tdeed_model,
        dataset=challenge_dataset,
        delta_frames_tolerance=1,
    )

    scored_videos = evaluator.get_scored_videos(
        batch_size=val_batch_size,
        use_snms=True,
        use_hflip=False,
        snms_params=dict(class_window=12, threshold=0.01),  # empirically found
    )

    evaluator.create_solution_file(
        scored_videos=scored_videos,
        zip_output_file_name=solution_archive_file_base_name,
    )
