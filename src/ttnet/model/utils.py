import torch


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
