from tvm import relax
from tvm.relax.testing import nn
from tvm.relax.op.nn import avg_pool2d


class AvgPool2d(nn.Module):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int | None = None,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, x: relax.Expr) -> relax.Var:
        assert self.count_include_pad and self.divisor_override is None, (
            self.count_include_pad and self.divisor_override
        )

        x = avg_pool2d(
            x,
            pool_size=self.kernel_size,
            strides=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            layout="NCHW",
        )
        return nn.emit(x)
