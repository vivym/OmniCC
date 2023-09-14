from tvm import relax
from tvm.relax.op import astype, reshape, permute_dims

from omni_cc import nn
from omni_cc.nn import functional as F
from .resnet import ResnetBlock2D, Upsample2D


class UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
    ):
        super().__init__()

        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([
                Upsample2D(out_channels, use_conv=True, out_channels=out_channels)
            ])
        else:
            self.upsamplers = None

    def forward(self, hidden_states: relax.Expr) -> relax.Var:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return nn.emit(hidden_states)


def get_up_block(
    up_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    prev_output_channel: int,
    temb_channels: int,
    add_upsample: int,
    resnet_eps: float,
    resnet_act_fn: str,
    attn_num_head_channels: int,
    resnet_groups: int | None = None,
    cross_attention_dim: int | None = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
) -> nn.Module:
    if up_block_type == "UpDecoderBlock2D":
        return UpDecoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )


class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_head_channels: int | None = None,
        norm_num_groups: int = 32,
        rescale_output_factor: float = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__()

        self.channels = channels
        self.num_heads = channels // num_head_channels if num_head_channels is not None else 1
        self.head_dim = num_head_channels

        self.group_norm = nn.GroupNorm(num_channels=channels, num_groups=norm_num_groups, eps=eps)

        # define q,k,v as linear layers
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)

        self.rescale_output_factor = rescale_output_factor
        self.proj_attn = nn.Linear(channels, channels)

        self._use_memory_efficient_attention_xformers = False
        self._attention_op = None

    def reshape_heads_to_batch_dim(self, x: relax.Expr) -> relax.Var:
        batch_size, seq_len, _ = x.struct_info.shape.values
        x = nn.emit(reshape(x, shape=[batch_size, seq_len, self.num_heads, self.head_dim]))
        x = nn.emit(permute_dims(x, [0, 2, 1, 3]))
        x = nn.emit(reshape(x, shape=[batch_size * self.num_heads, seq_len, self.head_dim]))
        return x

    def reshape_batch_dim_to_heads(self, x: relax.Expr) -> relax.Var:
        batch_size, seq_len, _ = x.struct_info.shape.values
        x = nn.emit(reshape(x, shape=[batch_size // self.num_heads, self.num_heads, seq_len, self.head_dim]))
        x = nn.emit(permute_dims(x, [0, 2, 1, 3]))
        x = nn.emit(reshape(x, shape=[batch_size // self.num_heads, seq_len, self.channels]))
        return x

    def forward(self, hidden_states: relax.Expr) -> relax.Var:
        batch_size, num_channels, height, width = hidden_states.struct_info.shape.values
        dtype = hidden_states.struct_info.dtype

        residual = hidden_states

        # group norm
        hidden_states = nn.emit(self.group_norm(hidden_states))
        hidden_states = nn.emit(reshape(hidden_states, shape=[batch_size, num_channels, -1]))
        hidden_states = nn.emit(permute_dims(hidden_states, [0, 2, 1]))

        # qkv projection
        query = nn.emit(self.query(hidden_states))
        key = nn.emit(self.key(hidden_states))
        value = nn.emit(self.value(hidden_states))

        # reshape qkv
        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        scale = relax.const(1.0 / (self.head_dim ** 0.5), dtype=dtype)

        # attention
        if self._use_memory_efficient_attention_xformers:
            assert NotImplementedError
        else:
            attention_scores = nn.emit(query @ nn.emit(permute_dims(key, [0, 2, 1])) * scale)
            attention_scores = astype(attention_scores, "float32")
            attention_probs = nn.emit(F.softmax(attention_scores, axis=-1))
            attention_probs = astype(attention_probs, dtype)
            hidden_states = nn.emit(attention_probs @ value)

        # reshape back
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)

        hidden_states = self.proj_attn(hidden_states)

        # reshape to spatial
        hidden_states = nn.emit(permute_dims(hidden_states, [0, 2, 1]))
        hidden_states = nn.emit(reshape(hidden_states, shape=[batch_size, num_channels, height, width]))

        # residual
        rescale_output_factor = relax.const(self.rescale_output_factor, dtype=dtype)
        hidden_states = nn.emit((hidden_states + residual) / rescale_output_factor)

        return hidden_states


class UNetMidBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attn_num_head_channels: int = 1,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()

        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    AttentionBlock(
                        in_channels,
                        num_head_channels=attn_num_head_channels,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=resnet_groups,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: relax.Expr, temb: relax.Expr | None = None) -> relax.Var:
        hidden_states = self.resnets[0](hidden_states, temb)

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states, temb)

        return nn.emit(hidden_states)
