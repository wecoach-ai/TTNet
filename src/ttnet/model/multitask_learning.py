import torch
import torch.nn as nn

from .loss import BallDetection, EventsSpotting, Segmentation
from .ttnet import TTNet
from .utils import create_target_ball


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
