import json
import logging
import pathlib

import numpy as np
import sklearn

from ..types import Config


def training_validation_data_separation(
    conf: Config,
) -> tuple[np.typing.NDArray[pathlib.Path], np.typing.NDArray[int], list[int] | None, list[int] | None]:
    training_data_dir = conf["dataset_dir"] / "training"
    training_games_list: list[pathlib.pathlib.Path] = []
    for path in training_data_dir.iterdir():
        if path.is_dir():
            training_games_list.append(path)

    events_information, events_labels = get_events_information(training_games_list, conf, "training")
    if not conf["validation"]:
        training_events_information = events_information
        training_events_labels = events_labels
        validation_events_information = None
        validation_events_labels = None

        return (
            training_events_information,
            validation_events_information,
            training_events_labels,
            validation_events_labels,
        )

    training_events_information, validation_events_information, training_events_labels, validation_events_labels = (
        sklearn.model_selection.train_test_split(
            events_information,
            events_labels,
            shuffle=True,
            test_size=conf["validation_size"],
            random_state=conf["seed"],
            stratify=events_labels,
        )
    )
    return training_events_information, validation_events_information, training_events_labels, validation_events_labels


def get_events_information(
    game_list: list[pathlib.Path], conf: Config, dataset_type: str
) -> tuple[tuple[list[pathlib.Path], np.typing.NDArray[int], None, pathlib.Path], list[int]]:
    number_frames_from_event = (conf["number_frame_sequence"] - 1) // 2

    annotations_dir = pathlib.Path(conf["dataset_dir"]) / dataset_type / "annotations"
    images_dir = pathlib.Path(conf["dataset_dir"]) / dataset_type / "images"

    events_information = []
    events_labels = []
    for game_name in game_list:
        ball_annotations_path = annotations_dir / game_name / "ball_markup.json"
        events_annotations_path = annotations_dir / game_name / "events_markup.json"
        with ball_annotations_path.open("r") as ball_fp, events_annotations_path.open("r") as events_fp:
            ball_annotations = json.load(ball_fp)
            events_annotations = json.load(events_fp)

        for frame_idx, name in events_annotations.items():
            frame_index = int(frame_idx)
            smooth_frame_indices = [frame_index]
            if name != "empty_event" and conf["smooth_labelling"]:
                smooth_frame_indices = [
                    index
                    for index in range(
                        frame_index - number_frames_from_event, frame_index + number_frames_from_event + 1
                    )
                ]

            for smooth_index in smooth_frame_indices:
                sub_smooth_frame_indices = [
                    index
                    for index in range(
                        smooth_index - number_frames_from_event, smooth_index + number_frames_from_event + 1
                    )
                ]
                image_path_list = [
                    images_dir / game_name / f"img_{sub_smooth_index:06d}.jpg"
                    for sub_smooth_index in sub_smooth_frame_indices
                ]
                last_frame_index = smooth_index + number_frames_from_event
                if f"{last_frame_index}" not in ball_annotations.keys():
                    logging.info(
                        f"smooth_index: {smooth_index} - no ball position for the frame idx {last_frame_index}"
                    )
                    continue
                ball_position_coordinates = np.array(
                    [ball_annotations[f"{last_frame_index}"]["x"], ball_annotations[f"{last_frame_index}"]["y"]],
                    dtype=int,
                )

                if ball_position_coordinates[0] < 0 or ball_position_coordinates[1] < 0:
                    continue

                segmentation_path = annotations_dir / game_name / "segmentation_masks" / f"{last_frame_index}.png"
                if not segmentation_path.is_file():
                    logging.info(f"smooth_index: {smooth_index} - The segmentation path {segmentation_path} is invalid")
                    continue
                event_class = conf["events_dict"][name]

                target_events = smooth_event_labelling(event_class, smooth_index, frame_index)
                events_information.append(
                    (image_path_list, ball_position_coordinates, target_events, segmentation_path)
                )

                if not target_events[0] and not target_events[1]:
                    event_class = 2
                events_labels.append(event_class)
    return events_information, events_labels


def smooth_event_labelling(event_class: int, smooth_index: int, event_frame_index: int) -> np.typing.NDArray[int]:
    target_events = np.zeros((2,))
    if event_class < 2:
        n = smooth_index - event_frame_index
        target_events[event_class] = np.cos(n * np.pi / 8)
        target_events[target_events < 0.01] = 0.0
    return target_events
