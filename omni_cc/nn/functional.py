from tvm import relax, te, tir
from tvm.relax.testing import nn
from tvm.relax.op import add, exp, flatten, log, multiply, reshape


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

    return x
