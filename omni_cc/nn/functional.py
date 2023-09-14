from tvm import relax, te, tir
from tvm.relax.testing import nn
from tvm.relax.op import add, exp, flatten, log, multiply, reshape
from tvm.relax.op.nn import softmax

__all__ = [
    "softplus",
    "softmax",
]


def softplus(x: relax.Expr, beta: float = 1, threshold: float = 20.0) -> relax.Var:
    input = x
    shape = x.struct_info.shape.values
    dtype = x.struct_info.dtype

    x = exp(multiply(relax.const(beta, dtype=dtype), x))
    x = log(add(relax.const(1, dtype=dtype), x))
    x = multiply(relax.const(1. / beta, dtype=dtype), x)

    def softplus_te(x: te.Tensor) ->  te.Tensor:
        def compute(i):
            return tir.Select(
                input[i] * beta > threshold,
                input[i],
                x[i],
            )

        return te.compute(x.shape, compute, name="softplus_select")

    x = flatten(x)
    x = nn.emit_te(softplus_te, x, primfunc_name_hint="softplus_select")
    x = reshape(x, shape=shape)

    return nn.emit(x)


def interpolate(
    x: relax.Expr,
    size: int | tuple[int, int] | None = None,
    scale_factor: float | tuple[float, float] | None = None,
    mode: str = "nearest",
    align_corners: bool | None = None,
    recompute_scale_factor: bool | None = None,
    antialias: bool = False,
) -> relax.Var:
    from tvm.relax.op.image import resize2d

    assert recompute_scale_factor is None
    assert antialias is False

    if size is None:
        assert scale_factor is not None
        shape = x.struct_info.shape
        assert isinstance(shape, relax.ShapeExpr)
        size = tuple(int(shape[i].value * scale_factor) for i in range(2, len(shape)))

    if mode.startswith("nearest"):
        mode = "nearest_neighbor"
    elif mode.startswith("bi"):
        mode = mode[2:]

    if mode == "nearest_neighbor":
        coord_trans = "asymmetric"
    elif align_corners:
        coord_trans = "align_corners"
    else:
        coord_trans = "half_pixel"

    return nn.emit(
        resize2d(
            x,
            size=size,
            layout="NCHW",
            method=mode,
            coordinate_transformation_mode=coord_trans,
        ),
    )
