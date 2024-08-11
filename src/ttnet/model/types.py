import typing

import torch

from .advanced_learning_strategies import MultiTaskLearning, UnbalanceLoss

TTNetSequentialModel: typing.TypeAlias = MultiTaskLearning | UnbalanceLoss

TTNetParallelModel: typing.TypeAlias = (
    torch.nn.parallel.DistributedDataParallel | torch.nn.DataParallel[TTNetSequentialModel]
)

TTNetModel: typing.TypeAlias = TTNetSequentialModel | TTNetParallelModel
