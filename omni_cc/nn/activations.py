from tvm import relax
from tvm.relax.testing import nn
from tvm.relax.op import sigmoid, tanh, multiply
from tvm.relax.op.nn import silu, gelu

from .functional import softplus
from .module import Module


class GELU(Module):
    def forward(self, x: relax.Expr) -> relax.Var:
        return nn.emit(gelu(x))


class QuickGELU(Module):
    def forward(self, x: relax.Expr) -> relax.Var:
        factor = relax.const(1.702, dtype=x.struct_info.dtype)
        return nn.emit(x * nn.emit(sigmoid(factor * x)))


class SiLU(Module):
    def forward(self, x: relax.Expr) -> relax.Var:
        return nn.emit(silu(x))


class Mish(Module):
    def forward(self, x: relax.Expr) -> relax.Var:
        return nn.emit(multiply(x, tanh(softplus(x))))
