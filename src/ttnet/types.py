import pathlib
import typing

import torch


class CLIParams(typing.TypedDict):
    seed: int
    saved_function: str
    number_frame_sequence: int
    model_arch: str
    drop_prob: float
    multitask_learning: bool
    local: bool
    event: bool
    segmentation: bool
    pretrained_path: pathlib.Path
    overwrite: bool
    dataset_dir: pathlib.Path
    working_dir: pathlib.Path
    validation: bool
    testing: bool
    validation_size: float
    smooth_labelling: bool
    num_sample: int
    num_worker: int
    batch_size: int
    print_frequency: int
    checkpoint_frequency: int
    sigma: float
    threshold_ball_position: float
    start_epoch: int
    num_epochs: int
    learning_rate: float
    minimum_learning_rate: float
    momentum: float
    weight_decay: float
    optimizer_type: str
    learning_rate_type: str
    learning_rate_factor: float
    learning_rate_step_size: int
    learning_rate_patience: int
    early_stop_patience: int
    freeze_global: bool
    freeze_local: bool
    freeze_event: bool
    freeze_segmentation: bool
    bce_weight: float
    global_weight: float
    local_weight: float
    event_weight: float
    segmentation_weight: float
    world_size: int
    rank: int
    dist_url: str
    dist_backend: str
    gpu_idx: int
    cuda: bool
    multiprocessing: bool
    evaluate: bool
    resume_path: pathlib.Path
    use_best_checkpoint: bool
    segmentation_threshold: float
    event_threshold: float
    save_test_output: bool
    video_path: pathlib.Path
    output_format: str
    show_image: bool
    save_demo_output: bool


class Config(CLIParams):
    device: torch.device
    number_gpu_per_node: int
    pin_memory: bool
    events_dict: dict[str, int]
    events_weights_loss: tuple[float, float]
    number_events: int
    original_frame_size: tuple[int, int]
    input_frame_size: tuple[int, int]
    tasks: set[str]
    tasks_loss_weight: list[float]
    freeze_modules_list: list[str]
    checkpoints_dir: pathlib.Path
    logs_dir: pathlib.Path
    saved_weight_file_name: pathlib.Path
    results_dir: pathlib.Path
    test_output_dir: pathlib.Path | None
    demo_output_dir: pathlib.Path | None
    distributed: bool


TTNetOptimizer: typing.TypeAlias = torch.optim.Adam | torch.optim.SGD
TTNetLearningRateScheduler: typing.TypeAlias = (
    torch.optim.lr_scheduler.StepLR | torch.optim.lr_scheduler.ReduceLROnPlateau
)
