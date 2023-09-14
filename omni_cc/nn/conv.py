from tvm import relax
from tvm.relax.testing import nn
from tvm.relax.op import add
from tvm.relax.op.nn import conv2d


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        dtype: str = "float32",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(
            (out_channels, in_channels // groups, *kernel_size),
            dtype=dtype,
            name="weight",
        )
        if bias:
            self.bias = nn.Parameter((out_channels,), dtype=dtype, name="bias")
        else:
            self.bias = None

    def forward(self, x: relax.Expr) -> relax.Var:
        x = nn.emit(
            conv2d(
                x,
                weight=self.weight,
                strides=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            ),
        )

        if self.bias is not None:
            x = nn.emit(add(x, self.bias))

        return x
