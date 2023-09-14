from tvm import relax
from omni_cc import nn, op
from omni_cc.nn import functional as F


class Upsample2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: int | None = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose

        conv = None
        if use_conv_transpose:
            conv = nn.ConvTranspose2d(
                in_channels, self.out_channels, kernel_size=4, stride=2, padding=1
            )
        elif use_conv:
            conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1)

        self.conv = conv

    def forward(self, x: relax.Expr, output_size: tuple[int, int] | None = None) -> relax.Var:
        if self.use_conv_transpose:
            return nn.emit(self.conv(x))

        if output_size is None:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        else:
            x = F.interpolate(x, size=output_size, mode="nearest")

        if self.use_conv:
            x = self.conv(x)

        return nn.emit(x)


class Downsample2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        use_conv: bool = False,
        out_channels: int | None = None,
        padding: int = 1,
    ):
        super().__init__()

        self.channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv
        self.padding = padding

        if use_conv:
            conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=3, stride=2, padding=padding)
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv = conv

    def forward(self, x: relax.Expr) -> relax.Var:
        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)

        x = self.conv(x)

        return nn.emit(x)


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: int | None = None,
        pre_norm: bool = True,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        time_embedding_norm: str = "default",  # default, scale_shift, ada_group
        output_scale_factor: float = 1.0,
        use_in_shortcut: bool | None = None,
        up: bool = False,
        down=False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: int | None = None,
    ):
        super().__init__()

        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm

        if groups_out is None:
            groups_out = groups

        if self.time_embedding_norm == "ada_group":
            self.norm1 = nn.AdaGroupNorm(temb_channels, in_channels, groups, eps=eps)
        else:
            self.norm1 = nn.GroupNorm(
                num_groups=groups, num_channels=in_channels, eps=eps)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = nn.Linear(temb_channels, out_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = nn.Linear(temb_channels, 2 * out_channels)
            elif self.time_embedding_norm == "ada_group":
                self.time_emb_proj = None
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm}.")
        else:
            self.time_emb_proj = None

        if self.time_embedding_norm == "ada_group":
            self.norm2 = nn.AdaGroupNorm(temb_channels, out_channels, groups_out, eps=eps)
        else:
            self.norm2 = nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps)

        self.dropout = nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = nn.Conv2d(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)

        if non_linearity == "swish":
            self.nonlinearity = nn.SiLU()
        elif non_linearity == "mish":
            self.nonlinearity = nn.Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()
        elif non_linearity == "gelu":
            self.nonlinearity = nn.GELU()

        self.upsample = self.downsample = None
        if self.up:
            self.upsample = Upsample2D(in_channels, use_conv=False)
        elif self.down:
            self.downsample = Downsample2D(in_channels, use_conv=False, padding=1, name="op")

        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels, conv_2d_out_channels, kernel_size=1, stride=1, padding=0, bias=conv_shortcut_bias
            )

    def forward(self, input_tensor: relax.Expr, temb: relax.Expr) -> relax.Var:
        hidden_states = input_tensor

        if self.time_embedding_norm == "ada_group":
            hidden_states = self.norm1(hidden_states, temb)
        else:
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))
            temb = op.expand_dims(temb, -1)
            temb = op.expand_dims(temb, -1)

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        if self.time_embedding_norm == "ada_group":
            hidden_states = self.norm2(hidden_states, temb)
        else:
            hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            dim = temb.struct_info.shape[1].value
            half_size = dim // 2
            stride = temb.struct_info.shape[2].value * temb.struct_info.shape[3].value
            print("stride", stride)
            scale = op.strided_slice(
                temb,
                axes=[1],
                begin=[0],
                end=[half_size],
                strides=[stride],
                assume_inbound=True,
            )
            shift = op.strided_slice(
                temb,
                axes=[1],
                begin=[half_size],
                end=[dim],
                strides=[stride],
                assume_inbound=True,
            )
            dtype = temb.struct_info.dtype
            hidden_states = hidden_states * (relax.const(1, dtype=dtype) + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_scale_factor = relax.const(
            self.output_scale_factor,
            dtype=hidden_states.struct_info.dtype,
        )
        output_tensor = (input_tensor + hidden_states) / output_scale_factor

        return nn.emit(output_tensor)
