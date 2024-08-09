import logging
import pathlib
import typing

import torch
from click import group, option, Choice
from click_option_group import optgroup

from .config import init_config
from .scheduler import trigger_training
from .types import CLIParams, Config


@group()
def cli() -> None:
    pass


@cli.command()
@option("--seed", type=int, default=2024, show_default=True, help="Seed for reproducing the result")
@option(
    "--saved-function",
    type=str,
    default="ttnet",
    show_default=True,
    help="Name used for saving logs, model state etc..",
)
@option(
    "--number-frame-sequence",
    type=int,
    default=9,
    show_default=True,
    help="Number of frames extracted for a particular event",
)
@optgroup.group("Model Configuration")
@optgroup.option("--model-arch", type=str, default="ttnet", show_default=True, help="Name of the model architecture")
@optgroup.option("--drop-prob", type=float, default=0.5, show_default=True, help="The dropout probability of the model")
@optgroup.option(
    "--multitask-learning/--no-multitask-learning",
    default=False,
    show_default=True,
    help="If true, the weights of different losses will be learnt (train). If false, a regular sum of different losses will be applied",
)
@optgroup.option(
    "--local/--no-local", default=False, show_default=True, help="If true, no local stage for ball detection"
)
@optgroup.option("--event/--no-event", default=False, show_default=True, help="If true, no event spotting detection")
@optgroup.option(
    "--segmentation/--no-segmentation", default=False, show_default=True, help="If true, no segmentation module"
)
@optgroup.option(
    "--pretrained-path",
    type=pathlib.Path,
    default=pathlib.Path.cwd(),
    show_default=True,
    help="The path of the pretrained checkpoint",
)
@optgroup.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="If true, the weights of the local stage will be overwritten by the global stage",
)
@optgroup.group("Dataloader And Running Configuration")
@optgroup.option(
    "--dataset-dir", type=pathlib.Path, default=pathlib.Path.cwd(), show_default=True, help="Dataset directory path"
)
@optgroup.option(
    "--working-dir", type=pathlib.Path, default=pathlib.Path.cwd(), show_default=True, help="Working directory path"
)
@optgroup.option(
    "--validation/--no-validation",
    default=False,
    show_default=True,
    help="If true, use all data for training, no validation set",
)
@optgroup.option(
    "--testing/--no-testing", default=False, show_default=True, help="If true, dont evaluate the model on the test set"
)
@optgroup.option("--validation-size", type=float, default=0.2, show_default=True, help="The size of validation set")
@optgroup.option(
    "--smooth-labelling/--no-smooth-labelling",
    default=False,
    show_default=True,
    help="If true, smoothly make the labels of event spotting",
)
@optgroup.option(
    "--num-sample", type=int, default=0, show_default=True, help="Take a subset of the dataset to run and debug"
)
@optgroup.option("--num-worker", type=int, default=4, show_default=True, help="Number of threads for loading data")
@optgroup.option(
    "--batch-size",
    type=int,
    default=8,
    show_default=True,
    help="This is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel",
)
@optgroup.option("--print-frequency", type=int, default=50, show_default=True, help="Printing Frequency")
@optgroup.option(
    "--checkpoint-frequency", type=int, default=2, show_default=True, help="Frequency of saving checkpoints"
)
@optgroup.option(
    "--sigma",
    type=float,
    default=1.0,
    show_default=True,
    help="Standard deviation of the 1D Gaussian for the ball position target",
)
@optgroup.option(
    "--threshold-ball-position",
    type=float,
    default=0.05,
    show_default=True,
    help="The lower threshold for the 1D Gaussian of the ball position target",
)
@optgroup.group("Training Strategy")
@optgroup.option("--start-epoch", type=int, default=1, show_default=True, help="The starting epoch index")
@optgroup.option("--num-epochs", type=int, default=30, show_default=True, help="The number of epochs to run")
@optgroup.option("--learning-rate", type=float, default=1e-3, show_default=True, help="The initial learning rate")
@optgroup.option(
    "--minimum-learning-rate",
    type=float,
    default=1e-7,
    show_default=True,
    help="The minimum learning rate during training",
)
@optgroup.option("--momentum", type=float, default=0.9, show_default=True, help="The momentum")
@optgroup.option("--weight-decay", type=float, default=1e-6, show_default=True, help="The weight decay")
@optgroup.option(
    "--optimizer-type",
    type=Choice(["SGD", "ADAM"], case_sensitive=False),
    default="SGD",
    show_default=True,
    help="The type of optimizer function",
)
@optgroup.option(
    "--learning-rate-type",
    type=Choice(["STEP", "REDUCEONPLATEAU"], case_sensitive=False),
    default="STEP",
    show_default=True,
    help="The type of learning rate scheduler",
)
@optgroup.option(
    "--learning-rate-factor", type=float, default=0.5, show_default=True, help="Factor to reduce the learning rate"
)
@optgroup.option(
    "--learning-rate-step-size",
    type=int,
    default=5,
    show_default=True,
    help="Step size of leanring rate when using --learning-rate-type=STEP",
)
@optgroup.option(
    "--learning-rate-patience",
    type=int,
    default=3,
    show_default=True,
    help="Patience of the learning rate when using --learning-rate-type=REDUCEONPLATEAU",
)
@optgroup.option(
    "--early-stop-patience",
    type=int,
    default=0,
    show_default=True,
    help="Early stopping the training process if performance is not improved within this value",
)
@optgroup.option(
    "--freeze-global/--no-freeze-global",
    default=False,
    show_default=True,
    help="If true, don't update/train weights for the global stage of ball detection",
)
@optgroup.option(
    "--freeze-local/--no-freeze-local",
    default=False,
    show_default=True,
    help="If true, don't update/train weights for the local stage of ball detection",
)
@optgroup.option(
    "--freeze-event/--no-freeze-event",
    default=False,
    show_default=True,
    help="If true, don't update/train weights for the event module",
)
@optgroup.option(
    "--freeze-segmentation/--no-freeze-segmentation",
    default=False,
    show_default=True,
    help="If true, don't update/train weights for the segmentation module",
)
@optgroup.group("Loss Weight Configuration")
@optgroup.option(
    "--bce-weight",
    type=float,
    default=0.5,
    show_default=True,
    help="The weight of BCE loss in segmentation module, the dice_loss weight = 1- bce_weight",
)
@optgroup.option(
    "--global-weight",
    type=float,
    default=1.0,
    show_default=True,
    help="The weight of loss of the global stage for ball detection",
)
@optgroup.option(
    "--local-weight",
    type=float,
    default=1.0,
    show_default=True,
    help="The weight of loss of the local stage for ball detection",
)
@optgroup.option(
    "--event-weight", type=float, default=1.0, show_default=True, help="The weight of loss of the event spotting module"
)
@optgroup.option(
    "--segmentation-weight",
    type=float,
    default=1.0,
    show_default=True,
    help="The weight of BCE loss in segmentation module",
)
@optgroup.group("Distributed Data Parallel Configuration")
@optgroup.option(
    "--world-size", type=int, default=-1, show_default=True, help="Number of nodes for distributed training"
)
@optgroup.option("--rank", type=int, default=-1, show_default=True, help="Node rank for distributed training")
@optgroup.option(
    "--dist-url",
    type=str,
    default="tcp://127.0.0.1:29500",
    show_default=True,
    help="URL used to set up distributed training",
)
@optgroup.option("--dist-backend", type=str, default="nccl", show_default=True, help="Distributed backend")
@optgroup.option("--gpu-idx", type=int, default=-1, show_default=True, help="GPU index to use")
@optgroup.option("--cuda/--no-cuda", default=False, show_default=True, help="If true, cuda is used")
@optgroup.option(
    "--multiprocessing/--no-multiprocessing",
    default=False,
    show_default=True,
    help="Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel training",
)
@optgroup.group("Evaluation Configuration")
@optgroup.option(
    "--evaluate/--no-evaluate", default=False, show_default=True, help="Only evaluate the model, don't train"
)
@optgroup.option(
    "--resume-path",
    type=pathlib.Path,
    default=pathlib.Path.cwd(),
    show_default=True,
    help="Path of the resumed checkpoint",
)
@optgroup.option(
    "--use-best-checkpoint/--no-use-best-checkpoint",
    default=False,
    show_default=True,
    help="If true, choose the best model on val set, otherwise choose the last model",
)
@optgroup.option(
    "--segmentation-threshold", type=float, default=0.5, show_default=True, help="Threshold of the segmentation output"
)
@optgroup.option(
    "--event-threshold", type=float, default=0.5, show_default=True, help="Threshold of the event spotting output"
)
@optgroup.option(
    "--save-test-output/--no-save-test-output",
    default=False,
    show_default=True,
    help="If true, the image of testing phase will be saved",
)
@optgroup.group("Demonstration Configuration")
@optgroup.option(
    "--video-path",
    type=pathlib.Path,
    default=pathlib.Path.cwd(),
    show_default=True,
    help="The path of the video that needs to demo",
)
@optgroup.option("--output-format", type=str, default="text", show_default=True, help="The type of the demo output")
@optgroup.option(
    "--show-image/--no-show-image", default=False, show_default=True, help="If true, show the image during demostration"
)
@optgroup.option(
    "--save-demo-output/--no-save-demo-output",
    default=False,
    show_default=True,
    help="If true, the image of demonstration phase will be saved",
)
def train(**params: typing.Unpack[CLIParams]) -> None:
    """
    Comamnd for training the ttnet model
    """
    conf: Config = init_config(params)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(module)s.py - %(funcName)s(), at Line %(lineno)d:%(levelname)s:\n%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(conf["logs_dir"] / f"logger_{conf['saved_function']}.log"),
        ],
    )

    if conf["multiprocessing"]:
        torch.multiprocessing.spawn(trigger_training, nprocs=conf["number_gpu_per_node"], args=(conf,))  # type: ignore
        return

    trigger_training(conf)
