import logging
import typing

from ..types import Config
from .ttnet import TTNet
from .multitask_learning import MultiTaskLearning
from .unbalance_loss import UnbalanceLoss


def model_factory(conf: Config) -> MultiTaskLearning | UnbalanceLoss:
    base_model_dict: dict[str, typing.Type[TTNet]] = {"ttnet": TTNet}
    base_model_class: typing.Type[TTNet] | None = base_model_dict.get(conf["model_arch"])

    if not base_model_class:
        logging.error("Model architecture not supported")
        raise NotImplementedError("Requested model architecture is not supported")

    base_model: TTNet = base_model_class(
        conf["drop_prob"],
        conf["tasks"],
        conf["input_frame_size"],
        conf["threshold_ball_position"],
        conf["number_frame_sequence"],
    )

    if conf["multitask_learning"]:
        return MultiTaskLearning(
            base_model,
            conf["tasks"],
            conf["number_events"],
            conf["events_weights_loss"],
            conf["input_frame_size"],
            conf["sigma"],
            conf["threshold_ball_position"],
            conf["device"],
        )

    return UnbalanceLoss(
        base_model,
        conf["tasks_loss_weight"],
        conf["events_weights_loss"],
        conf["input_frame_size"],
        conf["sigma"],
        conf["threshold_ball_position"],
        conf["device"],
    )
