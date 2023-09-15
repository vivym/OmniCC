import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from transformers.utils import cached_file

from omni_cc.utils.target_utils import parse_target

app = typer.Typer()


@app.command()
def build(
    pretrained_model_name_or_path: Annotated[
        str,
        typer.Option(
            "--pretrained_model_name_or_path",
            "-m",
            help=""
        ),
    ],
    output_dir: Annotated[
        str,
        typer.Option(
            "--output_dir",
            "-o",
            help="",
        ),
    ],
    cache_dir: Annotated[
        Optional[str],
        typer.Option(
            help="Path to a directory in which a downloaded pretrained model configuration should be cached.",
        ),
    ] = None,
    token: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, "
                "will use the token generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            ),
        ),
    ] = None,
    revision: Annotated[
        Optional[str],
        typer.Option(
            help="The specific model version to use.",
        ),
    ] = None,
    subfolder: Annotated[
        str,
        typer.Option(
            help=(
                "In case the relevant files are located inside a subfolder of the model repo on huggingface.co, "
                "you can specify the folder name here."
            ),
        ),
    ] = "",
    local_files_only: Annotated[
        bool,
        typer.Option(
            help="If True, don't try to download the model from huggingface.co.",
        ),
    ] = False,
    variant: Annotated[
        Optional[str],
        typer.Option(
            help="If specified load weights from `variant` filename, *e.g.* pytorch_model.<variant>.bin.",
        ),
    ] = None,
    target: Annotated[
        str,
        typer.Option(
            help="Compilation target."
        ),
    ] = "auto",
):
    tvm_target, tvm_target_kind = parse_target(target)

    model_index_path = cached_file(
        pretrained_model_name_or_path,
        filename="model_index.json",
        cache_dir=cache_dir,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
        subfolder=subfolder,
    )

    with open(model_index_path, "r") as f:
        model_index = json.load(f)

    pipeline_name = model_index["_class_name"]

    if pipeline_name == "StableDiffusionXLPipeline":
        from omni_cc.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline

        pipe = StableDiffusionXLPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tvm_target=tvm_target,
            tvm_target_kind=tvm_target_kind,
            cache_dir=cache_dir,
            token=token,
            revision=revision,
            subfolder=subfolder,
            local_files_only=local_files_only,
            variant=variant,
        )
    else:
        raise NotImplementedError(f"Unsupported pipeline: {pipeline_name}")

    Path(output_dir).mkdir(exist_ok=True)

    pipe.compile(output_dir=output_dir)


@app.command()
def tune():
    ...


@app.command()
def run():
    ...


if __name__ == "__main__":
    app()
