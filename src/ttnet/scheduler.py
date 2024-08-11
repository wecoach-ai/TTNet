import logging
import pathlib
import time

import numpy as np
import torch
import torch.utils
import tqdm

from .data.loader import create_training_validation_loader, create_testing_loader
from .model.factory import model_factory
from .model.types import TTNetModel
from .types import Config, TTNetOptimizer, TTNetLearningRateScheduler


class AverageMeter:
    def __init__(self, name: str, fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        return f"{self.name} {self.val:{self.fmt}} ({self.avg:{self.fmt}})"


class ProgressMeter(object):
    def __init__(self, number_batches: int, meters: list[AverageMeter], prefix: str = "") -> None:
        self.batch_fmtstr = self.get_batch_fmtstr(number_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch) -> None:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def get_message(self, batch) -> str:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return "\t".join(entries)

    def get_batch_fmtstr(self, number_batches: int) -> str:
        num_digits = len(str(number_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(number_batches) + "]"


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
        model: TTNetModel = model_factory(conf)

        if is_master_node:
            number_parameters = get_number_parameters(model)
            logging.info(f"Number of trained parameters of the model = {number_parameters}")

        optimizer = create_optimizer(conf, model)
        learning_rate_scheduler = create_learning_rate_scheduler(optimizer, conf)
        best_validation_loss = np.inf
        earlystop_count = 0
        is_best = False

        if conf["pretrained_path"].is_file():
            model = load_pretrained_model(model, conf["pretrained_path"], conf["gpu_idx"], conf["overwrite"])
            logging.info(f"Loaded pretrained model at {conf['pretrained_path']}")

        if conf["resume_path"].is_file():
            checkpoint = torch.load(
                conf["resume_path"], map_location="cpu" if conf["gpu_idx"] == -1 else f"cuda:{conf['gpu_idx']}"
            )
            if checkpoint["configs"].arch != conf["model_arch"]:
                raise ValueError(
                    f"Model architecture in checkpoint {checkpoint['configs'].arch} is different from the input {conf['model_arch']}"
                )

            logging.info(f"Loaded checkpoint {conf['resume_path']} (epoch {checkpoint['epoch']})")

            if hasattr(model, "module"):
                model.module.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint["state_dict"])

            optimizer.load_state_dict(checkpoint["optimizer"])
            learning_rate_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            best_validation_loss = checkpoint["best_val_loss"]
            earlystop_count = checkpoint["earlystop_count"]
            conf["start_epoch"] = checkpoint["epoch"] + 1

        logging.info("Loading dataset and getting dataloader")
        training_loader, validation_loader, training_sampler = create_training_validation_loader(conf)
        testing_loader = create_testing_loader(conf)

        logging.info(f"Number of batches in training set is {len(training_loader)}")
        logging.info(f"Number of batches in test set is {len(testing_loader)}")
        if validation_loader:
            logging.info(f"Number of batches in validation set is {len(validation_loader)}")

        if conf["evaluate"]:
            if not validation_loader:
                raise ValueError("Validation should not be None")
            validation_loss = evaluate_one_epoch(validation_loader, model, conf)
            logging.info(f"Evaluated validation loss: {validation_loss}")
            return

    except ValueError as e:
        logging.exception(e)
    finally:
        if writer:
            writer.close()  # type: ignore


def get_number_parameters(model: TTNetModel) -> int:
    if hasattr(model, "module"):
        return sum(parameter.numel() for parameter in model.module.parameters() if parameter.requires_grad)

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def create_optimizer(conf: Config, model: TTNetModel) -> TTNetOptimizer:
    training_parameters = (
        [parameter for parameter in model.module.parameters() if parameter.requires_grad]
        if hasattr(model, "module")
        else [parameter for parameter in model.parameters() if parameter.requires_grad]
    )

    if conf["optimizer_type"] == "SGD":
        return torch.optim.SGD(
            training_parameters, lr=conf["learning_rate"], momentum=conf["momentum"], weight_decay=conf["weight_decay"]
        )

    return torch.optim.Adam(training_parameters, lr=conf["learning_rate"], weight_decay=conf["weight_decay"])


def create_learning_rate_scheduler(optimizer: TTNetOptimizer, conf: Config) -> TTNetLearningRateScheduler:
    if conf["learning_rate_type"] == "STEP":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=conf["learning_rate_step_size"], gamma=conf["learning_rate_factor"]
        )

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=conf["learning_rate_factor"], patience=conf["learning_rate_patience"]
    )


def load_pretrained_model(model: TTNetModel, pretrained_path: pathlib.Path, gpu_idx: int, overwrite: bool):
    checkpoint = torch.load(pretrained_path, map_location="cpu" if gpu_idx == -1 else f"cuda:{gpu_idx}")

    model_state_dict = model.module.state_dict() if hasattr(model, "module") else model.model_state_dict()
    pretrained_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict}

    if overwrite:
        pretrained_dict = load_weights_local_stage(pretrained_dict)

    model_state_dict.update(pretrained_dict)
    model.module.load_state_dict(model_state_dict)

    return model


def load_weights_local_stage(pretrained_dict):
    local_weights_dict = {}
    for layer_name, v in pretrained_dict.items():
        if "ball_global_stage" in layer_name:
            layer_name_parts = layer_name.split(".")
            layer_name_parts[1] = "ball_local_stage"
            local_name = ".".join(layer_name_parts)
            local_weights_dict[local_name] = v

    return pretrained_dict | local_weights_dict


def evaluate_one_epoch(validation_loader, model, conf):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")

    progress = ProgressMeter(
        len(validation_loader),
        [batch_time, data_time, losses],
        prefix=f"Evaluate - Epoch: [{conf['start_epoch'] - 1}/{conf['num_epochs']}]",
    )
    model.eval()

    with torch.no_grad():
        start_time = time.time()
        for batch_index, (
            resized_images,
            original_ball_position_coordinates,
            global_ball_position_coordinates,
            target_events,
            target_segments,
        ) in enumerate(tqdm.tqdm(validation_loader)):
            data_time.update(time.time() - start_time)
            batch_size = resized_images.size(0)
            target_segments = target_segments.to(conf["device"], non_blocking=True)
            resized_images = resized_images.to(conf["device"], non_blocking=True).float()
            pred_ball_global, pred_ball_local, pred_events, pred_seg, local_ball_pos_xy, total_loss, _ = model(
                resized_images,
                original_ball_position_coordinates,
                global_ball_position_coordinates,
                target_events,
                target_segments,
            )

            if not conf["distributed"] and conf["gpu_idx"] == -1:
                total_loss = torch.mean(total_loss)

            reduced_loss = (
                reduce_tensor(total_loss.data, conf["world_size"]) if conf["distributed"] else total_loss.data
            )
            losses.update(to_python_float(reduced_loss), batch_size)

            torch.cuda.synchronize()
            batch_time.update(time.time() - start_time)

            if (batch_index + 1) % conf["print_freq"] == 0:
                logging.info(progress.get_message(batch_index))

            start_time = time.time()

    return losses.avg


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.reduce_op.SUM)
    return rt / world_size


def to_python_float(t):
    if hasattr(t, "item"):
        return t.item()
    else:
        return t[0]
