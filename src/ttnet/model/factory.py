import logging
import typing

import torch
import torch.nn as nn

from ..types import Config
from .advanced_learning_strategies import MultiTaskLearning, UnbalanceLoss
from .neural_networks import TTNet
from .types import TTNetSequentialModel, TTNetParallelModel, TTNetModel


def model_factory(conf: Config) -> TTNetModel:
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

    model: TTNetSequentialModel = (
        MultiTaskLearning(
            base_model,
            len(conf["tasks"]),
            conf["number_events"],
            conf["events_weights_loss"],
            conf["input_frame_size"],
            conf["sigma"],
            conf["threshold_ball_position"],
            conf["device"],
        )
        if conf["multitask_learning"]
        else UnbalanceLoss(
            base_model,
            conf["tasks_loss_weight"],
            conf["events_weights_loss"],
            conf["input_frame_size"],
            conf["sigma"],
            conf["threshold_ball_position"],
            conf["device"],
        )
    )

    processed_model: TTNetModel
    processed_model = make_data_parallel(model, conf)
    processed_model = freeze(model, conf["freeze_modules_list"])

    return processed_model


def make_data_parallel(model: TTNetSequentialModel, conf: Config) -> TTNetSequentialModel | TTNetParallelModel:
    if conf["distributed"] and conf["gpu_idx"] >= 0:
        torch.cuda.set_device(conf["gpu_idx"])
        model.cuda(conf["gpu_idx"])

        return nn.parallel.DistributedDataParallel(model, device_ids=[conf["gpu_idx"]], find_unused_parameters=True)

    if conf["distributed"]:
        model.cuda()

        return nn.parallel.DistributedDataParallel(model)

    if conf["gpu_idx"] >= 0:
        torch.cuda.set_device(conf["gpu_idx"])

        return model.cuda(conf["gpu_idx"])

    return nn.DataParallel(model).cuda()


def freeze(
    model: TTNetSequentialModel | TTNetParallelModel, freeze_modules_list: list[str]
) -> TTNetSequentialModel | TTNetParallelModel:
    for layer, parameter in model.named_parameters():
        parameter.requires_grad = True

        for freeze_module in freeze_modules_list:
            if freeze_module in layer:
                parameter.requires_grad = False
                break

    return model
