from torch.optim.lr_scheduler import (
    SequentialLR,
    LinearLR,
    CosineAnnealingLR,
    LRScheduler,
)


def get_lr_scheduler_with_warmup(
    optimizer, warm_up_steps, total_training_steps
) -> LRScheduler:

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warm_up_steps,
    )

    cosine_steps = total_training_steps - warm_up_steps

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warm_up_steps],
    )

    return scheduler
