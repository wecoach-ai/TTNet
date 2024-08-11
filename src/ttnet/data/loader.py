import pathlib

import cv2
import numpy as np
import torch
import turbojpeg

from ..types import Config
from .transformers import Compose, RandomCrop, RandomHFlip, RandomRotate
from .utils import training_validation_data_separation, get_events_information


class TTNetDataset(torch.utils.data.Dataset):
    def __init__(
        self, events_information, original_frame_size, input_frame_size, transformers=None, number_sample=None
    ):
        self.events_information = events_information
        self.width_original, self.height_orignal = original_frame_size[0], original_frame_size[1]
        self.width_input, self.height_input = input_frame_size[0], input_frame_size[1]
        self.width_resize_ratio = self.width_original / self.width_input
        self.height_resize_ratio = self.height_orignal / self.height_input
        self.transformers = transformers
        self.jpeg_reader = turbojpeg.TurboJPEG()

        if number_sample:
            self.events_information = self.events_information[:number_sample]

    def __len__(self) -> int:
        return len(self.events_information)

    def __getitem__(self, index):
        image_path_list, original_ball_position_coordinates, target_events, segmentation_path = self.events_information[
            index
        ]

        segmentation_image = cv2.cvtColor(cv2.imread(segmentation_path), cv2.COLOR_BGR2RGB)
        resized_images = []
        for image_path in image_path_list:
            with image_path.open("rb") as image_fp:
                resized_images.append(
                    cv2.resize(self.jpeg_reader.decode(image_fp.read(), 0), (self.width_input, self.height_input))
                )

        resized_images = np.dstack(resized_images)
        global_ball_position_coordinates = np.array(
            [
                original_ball_position_coordinates[0] / self.width_resize_ratio,
                original_ball_position_coordinates[1] / self.height_resize_ratio,
            ]
        )

        if self.transformers:
            resized_images, global_ball_position_coordinates, segmentation_image = self.transformers(
                resized_images, global_ball_position_coordinates, segmentation_image
            )

        original_ball_position_coordinates = np.array(
            [
                global_ball_position_coordinates[0] / (1.0 / self.width_resize_ratio),
                global_ball_position_coordinates[1] / (1.0 / self.height_resize_ratio),
            ]
        )

        self.check_ball_position(original_ball_position_coordinates, self.width_original, self.height_orignal)
        self.check_ball_position(global_ball_position_coordinates, self.width_input, self.height_input)

        resized_images = resized_images.transpose(2, 0, 1)
        target_segment = segmentation_image.transpose(2, 0, 1).astype(float)

        target_segment[target_segment < 75] = 0.0
        target_segment[target_segment >= 75] = 1.0

        return (
            resized_images,
            original_ball_position_coordinates.astype(int),
            global_ball_position_coordinates.astype(int),
            target_events,
            target_segment,
        )

    def check_ball_position(self, ball_position_coordinates, width, height) -> None:
        if not ((0 < ball_position_coordinates[0] < width) and (0 < ball_position_coordinates[1] < height)):
            ball_position_coordinates[0] = -1.0
            ball_position_coordinates[1] = -1.0


def create_training_validation_loader(conf: Config):
    training_transformer: Compose = Compose(
        [
            RandomCrop(max_reduction_percent=0.15, p=0.5),
            RandomHFlip(p=0.5),
            RandomRotate(rotation_angle_limit=10, p=0.5),
        ]
    )

    training_events_information, validation_events_information, *_ = training_validation_data_separation(conf)
    training_dataset = TTNetDataset(
        training_events_information,
        conf["original_frame_size"],
        conf["input_frame_size"],
        transformers=training_transformer,
        number_samples=conf["num_sample"],
    )
    training_sampler = (
        torch.utils.data.distributed.DistributedSampler(training_dataset) if conf["distributed"] else None
    )
    training_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=conf["batch_size"],
        shuffle=(training_sampler is None),
        pin_memory=conf["pin_memory"],
        num_workers=conf["num_worker"],
        sampler=training_sampler,
    )

    validation_loader = None
    if not conf["validation"]:
        validation_transformer = None
        validation_sampler = None
        validation_dataset = TTNetDataset(
            validation_events_information,
            conf["original_frame_size"],
            conf["input_frame_size"],
            transformers=validation_transformer,
            number_samples=conf["num_sample"],
        )
        if conf["distributed"]:
            validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset, shuffle=False)

        validation_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=conf["batch_size"],
            shuffle=False,
            pin_memory=conf["pin_memory"],
            num_workers=conf["num_worker"],
            sampler=validation_sampler,
        )

    return training_loader, validation_loader, training_sampler


def create_testing_loader(conf: Config):
    testing_transformers = None
    testing_data_dir = conf["dataset_dir"] / "testing"

    testing_games_list: list[pathlib.pathlib.Path] = []
    for path in testing_data_dir.iterdir():
        if path.is_dir():
            testing_games_list.append(path)

    testing_events_information, testing_events_labels = get_events_information(testing_games_list, conf, "testing")
    testing_dataset = TTNetDataset(
        testing_events_information,
        conf["original_frame_size"],
        conf["input_frame_size"],
        transformers=testing_transformers,
        number_samples=conf["num_samples"],
    )
    testing_sampler = None
    if conf["distributed"]:
        testing_sampler = torch.utils.data.distributed.DistributedSampler(testing_dataset)

    return torch.utils.data.DataLoader(
        testing_dataset,
        batch_size=conf["batch_size"],
        shuffle=False,
        pin_memory=conf["pin_memory"],
        num_workers=conf["num_worker"],
        sampler=testing_sampler,
    )
