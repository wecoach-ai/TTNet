import logging

import torch
import torch.utils

from .model.factory import model_factory
from .model import multitask_learning, unbalance_loss
from .types import Config


def trigger_training(conf: Config) -> None:
    if conf["gpu_idx"] >= 0:
        logging.info(f"Use GPU: {conf['gpu_idx']} for training")

    if conf["distributed"] and conf["multiprocessing"]:
        torch.distributed.init_process_group(
            backend=conf["dist_backend"], init_method=conf["dist_url"], world_size=conf["world_size"], rank=conf["rank"]
        )

    is_master_node: bool = not conf["distributed"] or (
        conf["distributed"] and conf["rank"] % conf["number_gpu_per_node"] == 0
    )

    writer = None  # type: ignore
    if is_master_node:
        logging.info(f"{conf=}")

        writer = torch.utils.tensorboard.SummaryWriter(log_dir=conf["logs_dir"] / "tensorboard")  # type: ignore

    try:
        model: multitask_learning.MultiTaskLearning | unbalance_loss.UnbalanceLoss = model_factory(conf)
        logging.info(model)
    except Exception:
        pass
    finally:
        if writer:
            writer.close()  # type: ignore