import torch
import torch.nn as nn


class BallDetection(nn.Module):
    def __init__(self, width: int, height: int, epsilon: float = 1e-9):
        super().__init__()

        self.width = width
        self.height = height
        self.epsilon = epsilon

    def forward(self, pred_ball_position: torch.Tensor, target_ball_position: torch.Tensor) -> torch.Tensor:
        x_pred = pred_ball_position[:, : self.width]
        y_pred = pred_ball_position[:, self.width :]

        x_target = target_ball_position[:, : self.width]
        y_target = target_ball_position[:, self.width :]

        loss_ball_x = -torch.mean(
            x_target * torch.log(x_pred + self.epsilon) + (1 - x_target) * torch.log(1 - x_pred + self.epsilon)
        )
        loss_ball_y = -torch.mean(
            y_target * torch.log(y_pred + self.epsilon) + (1 - y_target) * torch.log(1 - y_pred + self.epsilon)
        )

        return loss_ball_x + loss_ball_y


class EventsSpotting(nn.Module):
    def __init__(self, weights: tuple[int, int] = (1, 3), number_events: int = 2, epsilon: float = 1e-9):
        super().__init__()

        self.weights = torch.tensor(weights).view(1, 2)
        self.weights = self.weights / self.weights.sum()
        self.number_events = number_events
        self.epsilon = epsilon

    def forward(self, pred_events: torch.Tensor, target_events: torch.Tensor) -> torch.Tensor:
        self.weights = self.weights.cuda()
        return -torch.mean(
            self.weights
            * (
                target_events * torch.log(pred_events + self.epsilon)
                + (1.0 - target_events) * torch.log(1 - pred_events + self.epsilon)
            )
        )


class DICESmooth(nn.Module):
    def __init__(self, epsilon: float = 1e-9):
        super().__init__()

        self.epsilon = epsilon

    def forward(self, pred_seg: torch.Tensor, target_seg: torch.Tensor) -> torch.Tensor:
        return 1.0 - (
            (torch.sum(2 * pred_seg * target_seg) + self.epsilon)
            / (torch.sum(pred_seg) + torch.sum(target_seg) + self.epsilon)
        )


class BCE(nn.Module):
    def __init__(self, epsilon: float = 1e-9):
        super().__init__()

        self.epsilon = epsilon

    def forward(self, pred_seg: torch.Tensor, target_seg: torch.Tensor) -> torch.Tensor:
        return -torch.mean(
            target_seg * torch.log(pred_seg + self.epsilon) + (1 - target_seg) * torch.log(1 - pred_seg + self.epsilon)
        )


class Segmentation(nn.Module):
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()

        self.bce_criterion = BCE(epsilon=1e-9)
        self.dice_criterion = DICESmooth(epsilon=1e-9)
        self.bce_weight = bce_weight

    def forward(self, pred_seg: torch.Tensor, target_seg: torch.Tensor) -> torch.Tensor:
        target_seg = target_seg.float()
        loss_bce = self.bce_criterion(pred_seg, target_seg)
        loss_dice = self.dice_criterion(pred_seg, target_seg)
        loss_seg = (1 - self.bce_weight) * loss_dice + self.bce_weight * loss_bce
        return loss_seg
