import torch
import torch.nn as nn

from .ttnet import TTNet


class MultiTaskLearning(nn.Module):
    def __init__(
        self,
        base_model: TTNet,
        tasks: set[str],
        number_events: int,
        events_weights_loss: tuple[float, float],
        input_size: tuple[int, int],
        sigma: float,
        threshold_ball_position_mask: float,
        device: torch.device,
    ) -> None:
        super().__init__()
