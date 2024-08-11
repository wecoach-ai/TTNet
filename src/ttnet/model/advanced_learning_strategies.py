import torch
import torch.nn as nn

from .loss import BallDetection, EventsSpotting, Segmentation
from .neural_networks import TTNet
from .types import TTNetModel


def gaussian(position: torch.Tensor, mu_y: torch.Tensor, sigma: float) -> torch.Tensor:
    return torch.exp(-(((position - mu_y) / sigma) ** 2) / 2)


def create_target_ball(
    ball_position_coordinates: torch.Tensor,
    sigma: float,
    width: int,
    height: int,
    threshold_mask: float,
    device: torch.device,
) -> torch.Tensor:
    target_ball_position = torch.zeros((width + height,), device=device)
    if (width > ball_position_coordinates[0] > 0) and (height > ball_position_coordinates[1] > 0):
        x_position = torch.arange(0, width, device=device)
        target_ball_position[:width] = gaussian(x_position, ball_position_coordinates[0], sigma)

        y_position = torch.arange(0, height, device=device)
        target_ball_position[width:] = gaussian(y_position, ball_position_coordinates[1], sigma)

        target_ball_position[target_ball_position < threshold_mask] = 0.0

    return target_ball_position


def get_number_parameters(model: TTNetModel) -> int:
    if hasattr(model, "module"):
        return sum(parameter.numel() for parameter in model.module.parameters() if parameter.requires_grad)

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class MultiTaskLearning(nn.Module):
    def __init__(
        self,
        base_model: TTNet,
        number_tasks: int,
        number_events: int,
        events_weights_loss: tuple[float, float],
        input_size: tuple[int, int],
        sigma: float,
        threshold_ball_position_mask: float,
        device: torch.device,
    ) -> None:
        super().__init__()

        self.base_model = base_model
        self.log_vars = nn.Parameter(torch.zeros(number_tasks))
        self.width, self.height = input_size
        self.sigma = sigma
        self.threshold_ball_position_mask = threshold_ball_position_mask
        self.device = device
        self.ball_loss_criterion = BallDetection(self.width, self.height)
        self.event_loss_criterion = EventsSpotting(weights=self.events_weights_loss, number_events=number_events)
        self.segmentation_loss_criterion = Segmentation()

    def forward(
        self,
        resize_batch_input: torch.Tensor,
        org_ball_position_coordinates: torch.Tensor,
        global_ball_position_coordinates: torch.Tensor,
        target_events: torch.Tensor,
        target_segments: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[float]]:
        log_vars_index = 0
        pred_ball_global, pred_ball_local, pred_events, pred_segments, local_ball_position_coordinates = (
            self.base_model(resize_batch_input, org_ball_position_coordinates)
        )
        batch_size = pred_ball_global.size(0)
        target_ball_global = torch.zeros_like(pred_ball_global)
        for sample_index in range(batch_size):
            target_ball_global[sample_index] = create_target_ball(
                global_ball_position_coordinates[sample_index],
                self.sigma,
                self.width,
                self.height,
                self.threshold_ball_position_mask,
                self.device,
            )
        global_ball_loss = self.ball_loss_criterion(pred_ball_global, target_ball_global)
        total_loss = global_ball_loss / (torch.exp(2 * self.log_vars[log_vars_index])) + self.log_vars[log_vars_index]

        if pred_ball_local:
            log_vars_index += 1
            target_ball_local = torch.zeros_like(pred_ball_local)
            for sample_index in range(batch_size):
                target_ball_local[sample_index] = create_target_ball(
                    local_ball_position_coordinates[sample_index],
                    self.sigma,
                    self.width,
                    self.height,
                    self.threshold_ball_position_mask,
                    self.device,
                )
            local_ball_loss = self.ball_loss_criterion(pred_ball_local, target_ball_local)
            total_loss += (
                local_ball_loss / (torch.exp(2 * self.log_vars[log_vars_index])) + self.log_vars[log_vars_index]
            )

        if pred_events:
            log_vars_index += 1
            target_events = target_events.to(device=self.device)
            event_loss = self.event_loss_criterion(pred_events, target_events)
            total_loss += event_loss / (2 * torch.exp(self.log_vars[log_vars_index])) + self.log_vars[log_vars_index]

        if pred_segments:
            log_vars_index += 1
            seg_loss = self.seg_loss_criterion(pred_segments, target_segments)
            total_loss += seg_loss / (2 * torch.exp(self.log_vars[log_vars_index])) + self.log_vars[log_vars_index]

        return (
            pred_ball_global,
            pred_ball_local,
            pred_events,
            pred_segments,
            local_ball_position_coordinates,
            total_loss,
            self.log_vars.data.tolist(),
        )


class UnbalanceLoss(nn.Module):
    def __init__(
        self,
        base_model: TTNet,
        task_loss_weight: list[float],
        events_weights_loss: tuple[float, float],
        input_size: tuple[int, int],
        sigma: float,
        threshold_ball_position_mask: float,
        device: torch.device,
    ) -> None:
        super().__init__()

        self.base_model = base_model

        self.tasks_loss_weight = torch.tensor(task_loss_weight)
        self.tasks_loss_weight = self.tasks_loss_weight / self.tasks_loss_weight.sum()
        self.width, self.height = input_size
        self.sigma = sigma
        self.threshold_ball_position_mask = threshold_ball_position_mask
        self.device = device
        self.ball_loss_criterion = BallDetection(self.width, self.height)
        self.event_loss_criterion = EventsSpotting(
            weights=self.events_weights_loss, number_events=len(task_loss_weight)
        )
        self.segmentation_loss_criterion = Segmentation()

    def forward(
        self,
        resize_batch_input: torch.Tensor,
        org_ball_position_coordinates: torch.Tensor,
        global_ball_position_coordinates: torch.Tensor,
        target_events: torch.Tensor,
        target_segments: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        pred_ball_global, pred_ball_local, pred_events, pred_seg, local_ball_position_coordinates = self.model(
            resize_batch_input, org_ball_position_coordinates
        )
        batch_size = pred_ball_global.size(0)
        target_ball_global = torch.zeros_like(pred_ball_global)
        task_index = 0
        for sample_index in range(batch_size):
            target_ball_global[sample_index] = create_target_ball(
                global_ball_position_coordinates[sample_index],
                self.sigma,
                self.width,
                self.height,
                self.threshold_ball_position_mask,
                self.device,
            )
        global_ball_loss = self.ball_loss_criterion(pred_ball_global, target_ball_global)
        total_loss = global_ball_loss * self.tasks_loss_weight[task_index]

        if pred_ball_local:
            task_index += 1
            target_ball_local = torch.zeros_like(pred_ball_local)
            for sample_index in range(batch_size):
                target_ball_local[sample_index] = create_target_ball(
                    local_ball_position_coordinates[sample_index],
                    self.sigma,
                    self.width,
                    self.height,
                    self.threshold_ball_position_mask,
                    self.device,
                )
            local_ball_loss = self.ball_loss_criterion(pred_ball_local, target_ball_local)
            total_loss += local_ball_loss * self.tasks_loss_weight[task_index]

        if pred_events:
            task_index += 1
            target_events = target_events.to(device=self.device)
            event_loss = self.event_loss_criterion(pred_events, target_events)
            total_loss += event_loss * self.tasks_loss_weight[task_index]

        if pred_seg:
            task_index += 1
            seg_loss = self.seg_loss_criterion(pred_seg, target_segments)
            total_loss += seg_loss * self.tasks_loss_weight[task_index]

        return (
            pred_ball_global,
            pred_ball_local,
            pred_events,
            pred_seg,
            local_ball_position_coordinates,
            total_loss,
            None,
        )
