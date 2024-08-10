import typing

import torch

from .multitask_learning import MultiTaskLearning
from .unbalance_loss import UnbalanceLoss

TTNetSequentialModel: typing.TypeAlias = MultiTaskLearning | UnbalanceLoss

TTNetParallelModel: typing.TypeAlias = (
    torch.nn.parallel.DistributedDataParallel | torch.nn.DataParallel[TTNetSequentialModel]
)

TTNetModel: typing.TypeAlias = TTNetSequentialModel | TTNetParallelModel
