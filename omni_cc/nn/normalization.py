from tvm import relax
from tvm.relax.testing import nn
from tvm.relax.op import add, astype, multiply, expand_dims, strided_slice
from tvm.relax.op.nn import layer_norm, group_norm

from .activations import GELU, SiLU, Mish


class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()

        self.hidden_size = hidden_size
        self.eps = eps

        # force fp32 for now, TODO: mixed precision
        self.weight = nn.Parameter((hidden_size,), dtype="float32", name="weight")
        self.bias = nn.Parameter((hidden_size,), dtype="float32", name="bias")

    def forward(self, x: relax.Expr) -> relax.Var:
        x_dtype = x.struct_info.dtype
        if x_dtype != "float32":
            x = nn.emit(astype(x, "float32"))

        x = nn.emit(
            layer_norm(
                x,
                gamma=self.weight,
                beta=self.bias,
                axes=-1,
                epsilon=self.eps,
            )
        )

        if x_dtype != "float32":
            x = nn.emit(astype(x, x_dtype))

        return x


class GroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps

        # force fp32 for now, TODO: mixed precision
        self.weight = nn.Parameter((num_channels,), dtype="float32", name="weight")
        self.bias = nn.Parameter((num_channels,), dtype="float32", name="bias")

    def forward(self, x: relax.Expr) -> relax.Var:
        x_dtype = x.struct_info.dtype
        if x_dtype != "float32":
            x = nn.emit(astype(x, "float32"))

        ndim = len(x.struct_info.shape.values)

        x = nn.emit(
            group_norm(
                x,
                gamma=self.weight,
                beta=self.bias,
                num_groups=self.num_groups,
                channel_axis=1,
                axes=list(range(2, ndim)),
                epsilon=self.eps,
            )
        )

        if x_dtype != "float32":
            x = nn.emit(astype(x, x_dtype))

        return x


class AdaGroupNorm(nn.Module):
    """
    GroupNorm layer modified to incorporate timestep embeddings.
    """

    def __init__(
        self, embedding_dim: int, out_dim: int, num_groups: int, act_fn: str | None = None, eps: float = 1e-5
    ):
        super().__init__()

        self.num_groups = num_groups
        self.eps = eps

        self.act = None
        if act_fn == "swish":
            self.act = SiLU()
        elif act_fn == "mish":
            self.act = Mish()
        elif act_fn == "silu":
            self.act = SiLU()
        elif act_fn == "gelu":
            self.act = GELU()

        self.linear = nn.Linear(embedding_dim, out_dim * 2)

    def forward(self, x, emb):
        ndim = len(x.struct_info.shape.values)
        assert ndim == 2, ndim

        dtype = x.struct_info.dtype

        if self.act:
            emb = self.act(emb)

        emb = self.linear(emb)
        emb = expand_dims(emb, axis=-1)
        emb = expand_dims(emb, axis=-1)

        dim = x.struct_info.shape[1].value
        half_size = dim // 2
        scale = strided_slice(
            emb,
            axes=[1],
            begin=[0],
            end=[half_size],
            strides=[1],
            assume_inbound=True,
        )
        shift = strided_slice(
            emb,
            axes=[1],
            begin=[half_size],
            end=[dim],
            strides=[1],
            assume_inbound=True,
        )

        x = group_norm(
            x,
            gamma=relax.const(1, dtype=dtype),
            beta=relax.const(0, dtype=dtype),
            num_groups=self.num_groups,
            channel_axis=1,
            axes=[2, 3],
            epsilon=self.eps
        )
        x = nn.emit(
            add(
                multiply(
                    x,
                    add(
                        relax.const(1, dtype=scale.struct_info.dtype),
                        scale,
                    ),
                ),
                shift,
            ),
        )

        return x
