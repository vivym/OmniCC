from tvm import relax
from tvm.relax.testing import nn
from tvm.relax.op import linear


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: str = "float32",
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter((out_features, in_features), dtype=dtype, name="weight")
        if bias:
            self.bias = nn.Parameter((out_features,), dtype=dtype, name="bias")
        else:
            self.bias = None

    def forward(self, x: relax.Expr) -> relax.Var:
        x = nn.emit(
            linear(
                x,
                weight=self.weight,
                bias=self.bias,
            ),
        )

        return x
