import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pooling: bool = True) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batch_normalization = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pooling = pooling

        if self.pooling:
            self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.batch_normalization(self.convolution(x)))

        if self.pooling:
            return self.max_pooling(x)

        return x


class DeconvolutionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        middle_channels = in_channels // 4

        self.convolution_block_1 = nn.Conv2d(in_channels, middle_channels, kernel_size=1, stride=1, padding=0)
        self.batch_normalization_1 = nn.BatchNorm2d(middle_channels)
        self.relu = nn.ReLU()
        self.batch_normalization_transposed = nn.BatchNorm2d(middle_channels)
        self.convolution_transposed = nn.ConvTranspose2d(
            middle_channels, middle_channels, kernel_size=3, stride=1, padding=1, output_padding=1
        )
        self.convolution_block_2 = nn.Conv2d(middle_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_normalization_2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.batch_normalization_1(self.convolution_block_1(x)))
        x = self.relu(self.batch_normalization_transposed(self.convolution_transposed(x)))
        x = self.relu(self.batch_normalization_2(self.convolution_block_2(x)))
        return x


class BallDetection(nn.Module):
    def __init__(self, number_frame_sequence: int, dropout_probability: float) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(number_frame_sequence * 3, 64, kernel_size=1, stride=1, padding=0)
        self.batch_normalization = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.convolution_block_1 = ConvolutionBlock(64, 64)
        self.convolution_block_2 = ConvolutionBlock(64, 64)
        self.droupout_2d = nn.Dropout2d(p=dropout_probability)
        self.convolution_block_3 = ConvolutionBlock(64, 128)
        self.convolution_block_4 = ConvolutionBlock(128, 128)
        self.convolution_block_5 = ConvolutionBlock(128, 256)
        self.convolution_block_6 = ConvolutionBlock(256, 256)
        self.linear_transformation_1 = nn.Linear(in_features=2560, out_features=1792)
        self.linear_transformation_2 = nn.Linear(in_features=1792, out_features=896)
        self.linear_transformation_3 = nn.Linear(in_features=896, out_features=448)
        self.droupout_1d = nn.Dropout(p=dropout_probability)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.relu(self.batch_normalization(self.convolution(x)))
        out_block_2 = self.convolution_block_2(self.convolution_block_1(x))

        x = self.droupout_2d(out_block_2)
        out_block_3 = self.convolution_block_3(x)
        out_block_4 = self.convolution_block_4(out_block_3)

        x = self.droupout_2d(out_block_4)
        out_block_5 = self.convolution_block_5(out_block_4)
        features = self.convolution_block_6(out_block_5)

        x = self.droupout_2d(features)
        x = x.contiguous().view(x.size(0), -1)

        x = self.droupout_1d(self.relu(self.linear_transformation_1(x)))
        x = self.droupout_1d(self.relu(self.linear_transformation_2(x)))
        out = self.sigmoid(self.linear_transformation_3(x))

        return out, features, out_block_2, out_block_3, out_block_4, out_block_5


class EventsSpotting(nn.Module):
    def __init__(self, dropout_probability: float) -> None:
        super().__init__()
        self.convolution_block_1 = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0)
        self.batch_normalization = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.droupout = nn.Dropout2d(p=dropout_probability)
        self.convolution_block_2 = ConvolutionBlock(64, 64, pooling=False)
        self.linear_transformation_1 = nn.Linear(in_features=640, out_features=512)
        self.linear_transformation_2 = nn.Linear(in_features=512, out_features=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, global_features: torch.Tensor, local_features: torch.Tensor) -> torch.Tensor:
        input_event_spotting = torch.cat((global_features, local_features), dim=1)
        x = self.relu(self.batch_normalization(self.convolution_block_1(input_event_spotting)))
        x = self.droupout_2d(x)
        x = self.convolution_block_2(x)
        x = self.droupout_2d(x)
        x = self.convolution_block_2(x)
        x = self.droupout_2d(x)

        x = x.contiguous().view(x.size(0), -1)
        x = self.relu(self.linear_transformation_1(x))
        out = self.sigmoid(self.linear_transformation_2(x))

        return out


class Segmentation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.deconvolution_block_5 = DeconvolutionBlock(256, 128)
        self.deconvolution_block_4 = DeconvolutionBlock(128, 128)
        self.deconvolution_block_3 = DeconvolutionBlock(128, 64)
        self.deconvolution_block_2 = DeconvolutionBlock(64, 64)
        self.convolution_transposed = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=0, output_padding=0
        )
        self.relu = nn.ReLU()
        self.convolution_block_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.convolution_block_2 = nn.Conv2d(32, 3, kernel_size=2, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, out_block2: torch.Tensor, out_block3: torch.Tensor, out_block4: torch.Tensor, out_block5: torch.Tensor
    ) -> torch.Tensor:
        x = self.deconvolution_block_5(out_block5)
        x += out_block4

        x = self.deconvolution_block_4(x)
        x += out_block3

        x = self.deconvolution_block_3(x)
        x += out_block2

        x = self.deconvolution_block_2(x)
        x = self.relu(self.convolution_transposed(x))
        x = self.relu(self.convolution_block_1(x))
        out = self.sigmoid(self.convolution_block_2(x))

        return out


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
        self.ball_global_stage = BallDetection(number_frame_sequence, dropout_probability)
        self.ball_local_stage = BallDetection(number_frame_sequence, dropout_probability) if "local" in tasks else None
        self.events_spotting = EventsSpotting(dropout_probability) if "event" in tasks else None
        self.segmentation = Segmentation() if "segmentation" in tasks else None
        self.resize_width, self.resize_height = input_size
        self.threshold_ball_position_mask = threshold_ball_position_mask
        self.mean = torch.repeat_interleave(torch.tensor(mean).view(1, 3, 1, 1), repeats=9, dim=1)
        self.standard_deviation = torch.repeat_interleave(
            torch.tensor(standard_deviation).view(1, 3, 1, 1), repeats=9, dim=1
        )

    def forward(
        self, resize_batch_input: torch.Tensor, org_ball_position_coordinates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_ball_local, pred_events, pred_seg, local_ball_position_coordinates = None, None, None, None
        pred_ball_global, global_features, out_block2, out_block3, out_block4, out_block5 = self.ball_global_stage(
            self.normalize(resize_batch_input)
        )

        if self.ball_local_stage:
            input_ball_local, cropped_params = self.crop(resize_batch_input, pred_ball_global)
            local_ball_position_coordinates = self.get_groundtruth_local_ball_coordinates(
                org_ball_position_coordinates, cropped_params
            )
            pred_ball_local, local_features, *_ = self.ball_local_stage(self.normalize(input_ball_local))

            if self.events_spotting:
                pred_events = self.events_spotting(global_features, local_features)

        if self.segmentation:
            pred_seg = self.segmentation(out_block2, out_block3, out_block4, out_block5)

        return pred_ball_global, pred_ball_local, pred_events, pred_seg, local_ball_position_coordinates

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.mean.is_cuda:
            self.mean = self.mean.cuda()
            self.standard_deviation = self.standard_deviation.cuda()

        return (x / 255 - self.mean) / self.standard_deviation

    def crop(
        self, resize_batch_input: torch.Tensor, pred_ball_global: torch.Tensor
    ) -> tuple[torch.Tensor, list[list[bool, int, int, int, int, int, int]]]:
        original_height, original_width = 1080, 1920
        height_ratio = original_height / self.resize_height
        width_ratio = original_width / self.resize_width

        pred_ball_global_mask = pred_ball_global.clone().detach()
        pred_ball_global_mask[pred_ball_global_mask < self.threshold_ball_position_mask] = 0.0

        input_ball_local = torch.zeros_like(resize_batch_input)
        original_batch_input = F.interpolate(resize_batch_input, (original_height, original_width))

        cropped_params = []

        for index in range(resize_batch_input.size(0)):
            pred_ball_position_x = pred_ball_global_mask[index, : self.resize_width]
            pred_ball_position_y = pred_ball_global_mask[index, self.resize_width :]

            if (torch.sum(pred_ball_position_x) == 0.0) or (torch.sum(pred_ball_position_y) == 0.0):
                x_center = self.resize_width // 2
                y_center = self.resize_height // 2
                is_ball_detected = False
            else:
                x_center = torch.argmax(pred_ball_position_x)
                y_center = torch.argmax(pred_ball_position_y)
                is_ball_detected = True

            x_center = int(x_center * width_ratio)
            y_center = int(y_center * height_ratio)

            x_min, x_max, y_min, y_max = self.get_crop_params(
                x_center, y_center, self.resize_width, self.resize_height, original_width, original_height
            )

            cropped_height = y_max - y_min
            cropped_width = x_max - x_min
            x_pad, y_pad = 0, 0

            if cropped_height != self.resize_height or cropped_width != self.resize_width:
                x_pad = (self.resize_width - cropped_width) // 2
                y_pad = (self.resize_height - cropped_height) // 2
                input_ball_local[index, :, y_pad : (y_pad + cropped_height), x_pad : (x_pad + cropped_width)] = (
                    original_batch_input[index, :, y_min:y_max, x_min:x_max]
                )
            else:
                input_ball_local[index, :, :, :] = original_batch_input[index, :, y_min:y_max, x_min:x_max]

            cropped_params.append([is_ball_detected, x_min, x_max, y_min, y_max, x_pad, y_pad])

        return input_ball_local, cropped_params

    def get_crop_params(
        self,
        x_center: int,
        y_center: int,
        resize_width: int,
        resize_height: int,
        original_width: int,
        original_height: int,
    ) -> tuple[int, int, int, int]:
        x_min = max(0, x_center - resize_width // 2)
        y_min = max(0, y_center - resize_height // 2)

        x_max = min(original_width, x_min + resize_width)
        y_max = min(original_height, y_min + resize_height)

        return x_min, x_max, y_min, y_max

    def get_groundtruth_local_ball_coordinates(
        self,
        org_ball_position_coordinates: torch.Tensor,
        cropped_params: list[list[bool, int, int, int, int, int, int]],
    ) -> torch.Tensor:
        local_ball_position_coordinates = torch.zeros_like(org_ball_position_coordinates)

        for index, params in enumerate(cropped_params):
            is_ball_detected, x_min, x_max, y_min, y_max, x_pad, y_pad = params

            if is_ball_detected:
                local_ball_position_coordinates[index, 0] = max(
                    org_ball_position_coordinates[index, 0] - x_min + x_pad, -1
                )
                local_ball_position_coordinates[index, 1] = max(
                    org_ball_position_coordinates[index, 1] - y_min + y_pad, -1
                )
                if (
                    (local_ball_position_coordinates[index, 0] >= self.resize_width)
                    or (local_ball_position_coordinates[index, 1] >= self.resize_height)
                    or (local_ball_position_coordinates[index, 0] < 0)
                    or (local_ball_position_coordinates[index, 1] < 0)
                ):
                    local_ball_position_coordinates[index, 0] = -1
                    local_ball_position_coordinates[index, 1] = -1
                    continue

            local_ball_position_coordinates[index, 0] = -1
            local_ball_position_coordinates[index, 1] = -1
        return local_ball_position_coordinates
