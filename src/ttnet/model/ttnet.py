import torch.nn as nn


class TTNet(nn.Module):
    def __init__(
        self,
        dropout_probability: float,
        tasks: set[str],
        input_size: tuple[int, int],
        threshold_ball_position_mask: float,
        number_frame_sequence: int,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        standard_deviation: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
