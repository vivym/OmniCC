import os
import json

import tvm
from tvm import relax, tir
from transformers.utils import cached_file

from omni_cc import nn
from omni_cc.relax_models.autoencoder_kl import AutoencoderKL


class StableDiffusionXLPipeline:
    def __init__(self, mod: tvm.IRModule):
        super().__init__()

        self.mod = mod

        self.vae_decode = self.mod["vae_decode"]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        tvm_target: tvm.target.Target,
        tvm_target_kind: str,
        cache_dir: str | None = None,
        token: bool | str | None = None,
        revision: str | None = None,
        subfolder: str = "",
        local_files_only: bool = False,
        variant: str | None = None,
    ):
        # VAE
        vae_config_path = cached_file(
            pretrained_model_name_or_path,
            filename="config.json",
            cache_dir=cache_dir,
            token=token,
            revision=revision,
            local_files_only=local_files_only,
            subfolder=os.path.join(subfolder, "vae"),
        )

        with open(vae_config_path, "r") as f:
            vae_config = json.load(f)

        assert vae_config["_class_name"] == "AutoencoderKL"

        vae_args = {k: v for k, v in vae_config.items() if not k.startswith("_")}
        vae = AutoencoderKL(**vae_args)

        if variant is not None:
            params_file_name = f"diffusion_pytorch_model.{variant}.safetensors"
        else:
            params_file_name = "diffusion_pytorch_model.safetensors"

        vae_params_path = cached_file(
            pretrained_model_name_or_path,
            filename=params_file_name,
            cache_dir=cache_dir,
            token=token,
            revision=revision,
            local_files_only=local_files_only,
            subfolder=os.path.join(subfolder, "vae"),
        )

        bb = relax.BlockBuilder()

        with bb.function("vae_decode"):
            bsz = 1
            latent_channels = vae_config["latent_channels"]
            height = tir.Var("height", "int64")
            width = tir.Var("width", "int64")
            # height = 8
            # width = 8

            latents = nn.Placeholder((bsz, latent_channels, height, width), dtype="float32", name="latents")

            with bb.dataflow():
                sample = vae.decode(latents)
                params = [latents] + vae.parameters()
                gv = bb.emit_output(sample)

            bb.emit_func_output(gv, params=params)

        mod = bb.get()
        print(mod)

        return cls(mod)

    def compile(self, output_dir: str | None = None):
        ...
