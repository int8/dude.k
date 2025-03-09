import dataclasses
import enum
import json
import os
import time
from typing import Type, Optional

import torch
from contextlib import nullcontext

from torch.optim.lr_scheduler import (
    LRScheduler,
)
import torch.nn.functional as F

from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from dudek.config import DEFAULT_DEVICE
from dudek.data.team_bas import BASLabel, ActionLabel
from dudek.ml.data.tdeed import TeamTDeedDataset
from dudek.ml.model.tdeed.eval.two_heads import BASTeamTDeedEvaluator


from dudek.ml.model.tdeed.modules.tdeed import TDeedModule

from dudek.utils.ml import get_lr_scheduler_with_warmup


@dataclasses.dataclass
class TDeedLoss:
    total_loss: float
    ce_labels_loss: float
    mse_displacement_loss: float


def train(
    experiment_name: str,
    model: TDeedModule,
    labels_enum: Type[BASLabel | ActionLabel],
    train_dataset: TeamTDeedDataset,
    val_dataset: Optional[TeamTDeedDataset],
    eval_metric: str,
    nr_epochs: int,
    start_eval_epoch_nr: int,
    device=DEFAULT_DEVICE,
    foreground_weight: int = 5,
    train_batch_size: int = 4,
    val_batch_size: int = 8,
    acc_grad_iter: int = 1,
    warm_up_epochs: int = 1,
    save_as: Optional[str] = "best.pt",
    lr: float = 0.0004,
    loss_weights=None,
):

    assert eval_metric in ["loss", "map"], "eval_metric must be loss or map"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    hparam_dict = {
        "eval_metric": eval_metric,
        "foreground_weight": foreground_weight,
        "train_batch_size": train_batch_size,
        "val_batch_size": val_batch_size,
        "acc_grad_iter": acc_grad_iter,
        "start_eval_epoch_nr": start_eval_epoch_nr,
        "nr_epochs": nr_epochs,
        "displacement": train_dataset.displacement,
        "clip_length": len(train_dataset.clips[0].frames),
        "epoch_volume": len(train_dataset.clips),
        "horizontal_flip_proba": train_dataset.flip_proba,
        "crop_proba": train_dataset.crop_proba,
        "features_model_name": model.features_model_name,
        "temporal_shift_mode": model.temporal_shift_mode,
        "sgp_ks": model.sgp_ks,
        "sgp_k": model.sgp_k,
        "n_layers": model.n_layers,
        "camera_move_proba": train_dataset.camera_move_proba,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "optimizer": optimizer.__class__.__name__,
        "model": model.__class__.__name__,
        "train_dataset": train_dataset.__class__.__name__,
        "val_dataset": (
            val_dataset.__class__.__name__ if val_dataset is not None else None
        ),
        "device": device,
    }

    summary_writer = SummaryWriter(log_dir=f"runs/{experiment_name}_{time.time()}")
    summary_writer.add_text(
        "train/hyperparameters", json.dumps(hparam_dict, indent=4), global_step=0
    )

    train_data_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True
    )
    if val_dataset is not None:
        eval_data_loader = DataLoader(
            val_dataset, batch_size=val_batch_size, shuffle=True
        )
    optimizer_steps_per_epoch = len(train_data_loader) // acc_grad_iter

    scheduler = get_lr_scheduler_with_warmup(
        optimizer,
        warm_up_steps=optimizer_steps_per_epoch * warm_up_epochs,
        total_training_steps=(nr_epochs - warm_up_epochs) * optimizer_steps_per_epoch,
    )

    best_eval_metric = float("inf") if eval_metric == "loss" else 0.0
    evaluator = None
    if eval_metric == "map" and val_dataset is not None:
        evaluator = BASTeamTDeedEvaluator(
            model=model,
            dataset=val_dataset,
            delta_frames_tolerance=5,
        )

    for epoch_nr in range(nr_epochs):
        _go_through_epoch_train(
            model=model,
            labels_enum=labels_enum,
            data_loader=train_data_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            acc_grad_iter=acc_grad_iter,
            summary_writer=summary_writer,
            foreground_weight=foreground_weight,
            epoch_nr=epoch_nr,
            loss_weights=loss_weights or [1.5, 1],
        )

        if val_dataset is not None:
            if epoch_nr >= start_eval_epoch_nr:
                model.eval()
                if eval_metric == "loss":
                    eval_loss = _get_eval_loss(
                        model,
                        labels_enum=labels_enum,
                        data_loader=eval_data_loader,
                        foreground_weight=foreground_weight,
                        device=device,
                    )
                    if eval_loss.total_loss < best_eval_metric:
                        best_eval_metric = eval_loss.total_loss
                        if save_as:
                            torch.save(
                                model.state_dict(),
                                save_as,
                            )

                    summary_writer.add_scalar(
                        "eval/total_loss", eval_loss.total_loss, epoch_nr
                    )

                    summary_writer.add_scalar(
                        "eval/ce_labels_loss", eval_loss.ce_labels_loss, epoch_nr
                    )

                    summary_writer.add_scalar(
                        "eval/mse_displacement_loss",
                        eval_loss.mse_displacement_loss,
                        epoch_nr,
                    )
                elif eval_metric == "map":
                    maps, map_mine = evaluator.eval(batch_size=val_batch_size)
                    if maps["mAP"] > best_eval_metric:
                        best_eval_metric = maps["mAP"]
                        if save_as:
                            torch.save(
                                model.state_dict(),
                                save_as,
                            )

                    summary_writer.add_scalar(
                        "eval/map",
                        maps["mAP"],
                        epoch_nr,
                    )

                    summary_writer.add_scalar(
                        "eval/map_mine",
                        map_mine,
                        epoch_nr,
                    )

                model.train()
    return model


def _get_eval_loss(
    model: TDeedModule,
    labels_enum: Type[BASLabel | ActionLabel],
    data_loader: DataLoader[TeamTDeedDataset],
    foreground_weight: int = 5,
    device=DEFAULT_DEVICE,
) -> TDeedLoss:
    epoch_loss_c, epoch_loss_d = _go_through_epoch_eval(
        model,
        labels_enum=labels_enum,
        data_loader=data_loader,
        foreground_weight=foreground_weight,
        device=device,
    )
    epoch_loss = epoch_loss_c.detach().item()
    epoch_loss += epoch_loss_d.detach().item()

    return TDeedLoss(
        total_loss=epoch_loss / len(data_loader),
        ce_labels_loss=epoch_loss_c.detach().item() / len(data_loader),
        mse_displacement_loss=epoch_loss_d.detach().item() / len(data_loader),
    )


def _go_through_epoch_train(
    model: TDeedModule,
    labels_enum: Type[BASLabel | ActionLabel],
    data_loader: DataLoader[TeamTDeedDataset],
    scheduler: LRScheduler = None,
    scaler: torch.cuda.amp.GradScaler = None,
    optimizer: torch.optim.Optimizer = None,
    acc_grad_iter: int = None,
    epoch_nr: int = None,
    foreground_weight: int = 5,
    device=DEFAULT_DEVICE,
    summary_writer: SummaryWriter = None,
    loss_weights=None,
):
    return _go_through_epoch(
        model=model,
        labels_enum=labels_enum,
        data_loader=data_loader,
        evaluate=False,
        scheduler=scheduler,
        scaler=scaler,
        optimizer=optimizer,
        acc_grad_iter=acc_grad_iter,
        epoch_nr=epoch_nr,
        foreground_weight=foreground_weight,
        device=device,
        summary_writer=summary_writer,
        loss_weights=loss_weights or [1.5, 1],
    )


def _go_through_epoch_eval(
    model: TDeedModule,
    labels_enum: Type[BASLabel | ActionLabel],
    data_loader: DataLoader[TeamTDeedDataset],
    foreground_weight: int = 5,
    device: str = DEFAULT_DEVICE,
    loss_weights=None,
):

    return _go_through_epoch(
        model=model,
        labels_enum=labels_enum,
        data_loader=data_loader,
        evaluate=True,
        foreground_weight=foreground_weight,
        device=device,
        loss_weights=loss_weights or [1.5, 1],
    )


def _go_through_epoch(
    model: TDeedModule,
    labels_enum: Type[BASLabel | ActionLabel],
    data_loader: DataLoader[TeamTDeedDataset],
    evaluate: bool = False,
    scheduler: LRScheduler = None,
    scaler: torch.cuda.amp.GradScaler = None,
    optimizer: torch.optim.Optimizer = None,
    acc_grad_iter: int = None,
    epoch_nr: int = None,
    foreground_weight: float = 5,
    device=DEFAULT_DEVICE,
    summary_writer: SummaryWriter = None,
    loss_weights=None,
):
    loss_weights = loss_weights or [1.5, 1]
    if not evaluate:
        optimizer.zero_grad()

    epoch_loss_c = 0.0
    epoch_loss_d = 0.0

    class_weights = torch.FloatTensor(
        [1] + [foreground_weight for _ in labels_enum] * 2
    ).to(device)

    batch_idx = 0
    with torch.no_grad() if evaluate else nullcontext():
        for batch in tqdm(data_loader, total=len(data_loader)):

            clip_tensor = batch["clip_tensor"]
            label = batch["labels_vector"]
            labels_displacement_vector = batch["labels_displacement_vector"]

            label = (
                label.flatten()
                if len(label.shape) == 2
                else label.view(-1, label.shape[-1])
            )

            with torch.amp.autocast(device):
                pred_dict, y = model(clip_tensor, y=label, inference=evaluate)

                pred = pred_dict["im_feat"]

                if "displ_feat" in pred_dict.keys():
                    pred_displacement = pred_dict["displ_feat"]

                loss = 0.0
                loss_c = 0.0

                predictions = pred.reshape(-1, (2 * len(labels_enum)) + 1)
                loss_c += F.cross_entropy(predictions, label, weight=class_weights)

                epoch_loss_c += loss_c * loss_weights[0]
                loss += loss_c * loss_weights[0]

                loss_d = F.mse_loss(
                    pred_displacement,
                    labels_displacement_vector,
                    reduction="none",
                )

                loss_d = loss_d.mean()
                epoch_loss_d += loss_d * loss_weights[1]
                loss += loss_d * loss_weights[1]

            if not evaluate:
                if summary_writer:
                    summary_writer.add_scalar(
                        "train/loss",
                        loss.detach().item(),
                        epoch_nr * len(data_loader) + batch_idx,
                    )
                    summary_writer.add_scalar(
                        "train/learning_rate",
                        optimizer.param_groups[0]["lr"],
                        epoch_nr * len(data_loader) + batch_idx,
                    )
                else:
                    print("train/loss", loss.detach().item())
            if not evaluate:
                _optim_step(
                    scaler=scaler,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss=loss,
                    backward_only=(batch_idx + 1) % acc_grad_iter != 0,
                )
            batch_idx += 1

    return epoch_loss_c, epoch_loss_d


def _optim_step(scaler, optimizer, scheduler, loss, backward_only=False):
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()

    if not backward_only:
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
