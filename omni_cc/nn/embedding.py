from tvm import relax
from tvm.relax.testing import nn
from tvm.relay.op import reshape, take

from .module import Module


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, dtype: str = "float32"):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter((num_embeddings, embedding_dim), dtype=dtype, name="weight")

    def forward(self, x: relax.Expr) -> relax.Var:
        x_shape = x.struct_info.shape.values
        x = nn.emit(reshape(x, shape=[-1]))
        x = nn.emit(take(self.weight, x, axis=0))
        x = nn.emit(reshape(x, shape=[*x_shape, self.embedding_dim]))
        return x
