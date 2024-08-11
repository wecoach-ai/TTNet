import typing

from .transformers import RandomCrop, RandomHFlip, RandomRotate


Transformer: typing.TypeAlias = RandomCrop | RandomHFlip | RandomRotate
