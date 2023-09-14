import unittest

from tvm import relax

from omni_cc import nn
from omni_cc.relax_models.autoencoder_kl import AutoencoderKL


class TestAutoencoderKL(unittest.TestCase):
    def test_decode(self):
        bb = relax.BlockBuilder()

        with bb.function("vae_decode"):
            vae = AutoencoderKL(
                in_channels=3,
                out_channels=3,
                down_block_types=(
                    "DownEncoderBlock2D",
                    "DownEncoderBlock2D",
                    "DownEncoderBlock2D",
                    "DownEncoderBlock2D",
                ),
                up_block_types=(
                    "UpDecoderBlock2D",
                    "UpDecoderBlock2D",
                    "UpDecoderBlock2D",
                    "UpDecoderBlock2D",
                ),
                block_out_channels=(
                    128,
                    256,
                    512,
                    512,
                ),
                layers_per_block=2,
                act_fn="silu",
                latent_channels=4,
                norm_num_groups=32,
                sample_size=1024,
                scaling_factor=0.13025,
                force_upcast=True,
            )

            latents = nn.Parameter((1, 4, 64, 64), dtype="float32", name="latents")

            with bb.dataflow():
                sample = vae.decode(latents)
                params = [latents] + vae.parameters()
                gv = bb.emit_output(sample)

            bb.emit_func_output(gv, params=params)

        mod = bb.get()
        gv = mod.get_global_var("vae_decode")
        bb.update_func(gv, mod[gv].with_attr("num_input", 1))

        print(mod)


if __name__ == "__main__":
    unittest.main()
