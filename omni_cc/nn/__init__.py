from typing import Iterable

from tvm import relax
from tvm.relax.testing.nn import (
    Placeholder,
    Parameter,
    Module,
    Sequential,
    ReLU,
    LogSoftmax,
    Linear,
    emit,
    emit_te,
    emit_checkpoint,
    emit_checkpoint_sequential,
    init_params,
)

from .activations import GELU, QuickGELU, SiLU, Mish
from .conv import Conv2d, ConvTranspose2d
from .embedding import Embedding
from .normalization import LayerNorm, GroupNorm, AdaGroupNorm
from .pooling import AvgPool2d


class ModuleList(Module):
    def __init__(self, modules: list[Module]):
        self.modules = modules

    def __iter__(self) -> Iterable[Module]:
        return iter(self.modules)

    def __getitem__(self, idx) -> Module:
        return self.modules[idx]

    def __len__(self) -> int:
        return len(self.modules)

    def forward(self, x: relax.Expr) -> relax.Var:
        for module in self.modules:
            x = module(x)
        return x
