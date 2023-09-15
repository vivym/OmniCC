from tvm import relax
from tvm.relax.testing import nn
from tvm.relax.op import add, reshape
from tvm.relax.op.nn import conv2d, conv2d_transpose

from .module import Module


class Conv2d(Module):
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
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(
            (out_channels, in_channels // groups) + kernel_size,
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
            bias = nn.emit(reshape(self.bias, shape=(1, -1, 1, 1)))
            x = nn.emit(add(x, bias))

        return x


class ConvTranspose2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        output_padding: int | tuple[int, int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int | tuple[int, int] = 1,
        dtype: str = "float32",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation

        self.weight = nn.Parameter(
            (in_channels, out_channels // groups) + kernel_size,
            dtype=dtype,
            name="weight",
        )
        if bias:
            self.bias = nn.Parameter((out_channels,), dtype=dtype, name="bias")
        else:
            self.bias = None

    def forward(self, x: relax.Expr) -> relax.Var:
        x = nn.emit(
            conv2d_transpose(
                x,
                weight=self.weight,
                strides=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                data_layout="NCHW",
                kernel_layout="IOHW",
            ),
        )

        if self.bias is not None:
            bias = nn.emit(reshape(self.bias, shape=(1, -1, 1, 1)))
            x = nn.emit(add(x, bias))

        return x
