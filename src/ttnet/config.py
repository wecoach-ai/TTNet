import pathlib
import random
import warnings

import numpy as np
import torch

from .types import CLIParams, Config


def generate_config(params: CLIParams) -> Config:
    tasks: set[str] = {"global", "local", "event", "segmentation"}
    if not params["local"]:
        tasks.discard("local")
        tasks.discard("event")

    if not params["event"]:
        tasks.discard("event")

    if not params["segmentation"]:
        tasks.discard("segmentation")

    loss_weights: dict[str, float] = {
        "global": params["global_weight"],
        "local": params["local_weight"],
        "event": params["event_weight"],
        "segmentation": params["segmentation_weight"],
    }
    freeze_modules: dict[str, str] = {
        "freeze_global": "ball_global_stage",
        "freeze_local": "ball_local_stage",
        "freeze_event": "events_spotting",
        "freeze_segmentation": "segmentation",
    }

    checkpoints_dir: pathlib.Path = params["working_dir"] / "checkpoints" / params["saved_function"]
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logs_dir: pathlib.Path = params["working_dir"] / "logs" / params["saved_function"]
    logs_dir.mkdir(parents=True, exist_ok=True)

    saved_weight_file_name: pathlib.Path = (
        checkpoints_dir / f"{params['saved_function']}{'_best.pth' if params['use_best_checkpoint'] else '.pth'}"
    )

    results_dir: pathlib.Path = params["working_dir"] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    test_output_dir: pathlib.Path | None = None
    if params["save_test_output"]:
        test_output_dir = results_dir / params["saved_function"]
        test_output_dir.mkdir(parents=True, exist_ok=True)

    demo_output_dir: pathlib.Path | None = None
    if params["save_demo_output"]:
        demo_output_dir = results_dir / "demo" / params["saved_function"]
        demo_output_dir.mkdir(parents=True, exist_ok=True)

    seed_model(params["seed"])

    if params["gpu_idx"]:
        warnings.warn("You have chosen a specific GPU. This will completely disable data parallelism.")

    if params["multiprocessing"]:
        params["world_size"] *= torch.cuda.device_count()

    return Config(
        **params,
        device=torch.device("cuda" if params["cuda"] else "cpu"),
        number_gpu_per_node=torch.cuda.device_count(),
        pin_memory=True,
        events_dict={"bounce": 0, "net": 1, "empty_event": 2},
        events_weights_loss=(1.0, 3.0),
        number_events=2,
        original_frame_size=(1920, 1080),
        input_frame_size=(320, 128),
        tasks=tasks,
        tasks_loss_weight=[loss_weights[task] for task in tasks],
        freeze_modules_list=[value for key, value in freeze_modules.items() if params[key]],  # type: ignore
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        saved_weight_file_name=saved_weight_file_name,
        results_dir=results_dir,
        test_output_dir=test_output_dir,
        demo_output_dir=demo_output_dir,
        distributed=params["world_size"] > 1 or params["multiprocessing"],
    )


def seed_model(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
