import logging

import torch
from torch.utils.tensorboard import SummaryWriter

from .model.factory import model_factory
from .types import Config


def trigger_training(conf: Config) -> None:
    if conf["gpu_idx"] >= 0:
        logging.info(f"Use GPU: {conf['gpu_idx']} for training")

    if conf["distributed"] and conf["multiprocessing"]:
        torch.distributed.init_process_group(
            backend=conf["dist_backend"], init_method=conf["dist_url"], world_size=conf["world_size"], rank=conf["rank"]
        )

    is_master_node = not conf["distributed"] or (
        conf["distributed"] and conf["rank"] % conf["number_gpu_per_node"] == 0
    )

    writer = None
    if is_master_node:
        logging.info(f"{conf=}")

        writer = SummaryWriter(log_dir=conf["logs_dir"] / "tensorboard")  # type: ignore

    try:
        model = model_factory(conf)
        logging.info(model)
    except Exception:
        pass
    finally:
        if writer:
            writer.close()  # type: ignore
