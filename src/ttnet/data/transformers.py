import random

import cv2
import numpy as np

from .types import Transformer


class Compose:
    def __init__(self, transforms: list[Transformer], p: float = 1.0):  # TODO:- Add container element type
        self.transforms = transforms
        self.p = p

    def __call__(self, images, ball_position_coordinates, segmentation_image):
        if random.random() <= self.p:
            for transform in self.transforms:
                images, ball_position_coordinates, segmentation_image = transform(
                    images, ball_position_coordinates, segmentation_image
                )

        return images, ball_position_coordinates, segmentation_image


class RandomCrop:
    def __init__(self, max_reduction_percent: float = 0.15, p: float = 0.5, interpolation: int = cv2.INTER_LINEAR):
        self.max_reduction_percent = max_reduction_percent
        self.p = p
        self.interpolation = interpolation

    def __call__(self, images, ball_position_coordinates, segmentation_images):
        if random.random() <= self.p:
            height, width, _ = images.shape
            remain_percent = random.uniform(1.0 - self.max_reduction_percent, 1.0)
            new_width = remain_percent * width
            min_x = int(random.uniform(0, width - new_width))
            max_x = int(min_x + new_width)
            width_ratio = width / new_width

            new_height = remain_percent * height
            min_y = int(random.uniform(0, height - new_height))
            max_y = int(new_height + min_y)
            height_ratio = height / new_height

            images = images[min_y:max_y, min_x:max_x, :]
            images = cv2.resize(images, (width, height), interpolation=self.interpolation)

            segmentation_images_height, segmentation_images_width, _ = segmentation_images.shape

            if (segmentation_images_height != height) or (segmentation_images_width != width):
                segmentation_images = cv2.resize(segmentation_images, (width, height), interpolation=self.interpolation)

            segmentation_images = segmentation_images[min_y:max_y, min_x:max_x, :]

            segmentation_images = cv2.resize(
                segmentation_images,
                (segmentation_images_width, segmentation_images_height),
                interpolation=self.interpolation,
            )

            ball_position_coordinates = np.array(
                [
                    (ball_position_coordinates[0] - min_x) * width_ratio,
                    (ball_position_coordinates[1] - min_y) * height_ratio,
                ]
            )

        return images, ball_position_coordinates, segmentation_images


class RandomHFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, images, ball_position_coordinates, segmentation_images):
        if random.random() <= self.p:
            _, width, _ = images.shape
            images = cv2.flip(images, 1)
            segmentation_images = cv2.flip(segmentation_images, 1)

            ball_position_coordinates[0] = width - ball_position_coordinates[0]

        return images, ball_position_coordinates, segmentation_images


class RandomRotate:
    def __init__(self, rotation_angle_limit: int = 15, p: float = 0.5):
        self.rotation_angle_limit = rotation_angle_limit
        self.p = p

    def __call__(self, images, ball_position_coordinates, segmentation_images):
        if random.random() <= self.p:
            random_angle = random.uniform(-self.rotation_angle_limit, self.rotation_angle_limit)

            height, width, _ = images.shape
            center = (int(width / 2), int(height / 2))
            rotate_matrix = cv2.getRotationMatrix2D(center, random_angle, 1.0)
            images = cv2.warpAffine(images, rotate_matrix, (width, height), flags=cv2.INTER_LINEAR)

            ball_position_coordinates = rotate_matrix.dot(
                np.array([ball_position_coordinates[0], ball_position_coordinates[1], 1.0]).T
            )

            segmentation_height, segmentation_width, _ = segmentation_images.shape
            if (segmentation_height != height) or (segmentation_width != width):
                seg_center = (int(segmentation_width / 2), int(segmentation_height / 2))
                seg_rotate_matrix = cv2.getRotationMatrix2D(seg_center, random_angle, 1.0)
            else:
                seg_rotate_matrix = rotate_matrix
            segmentation_images = cv2.warpAffine(
                segmentation_images,
                seg_rotate_matrix,
                (segmentation_width, segmentation_height),
                flags=cv2.INTER_LINEAR,
            )

        return images, ball_position_coordinates, segmentation_images
