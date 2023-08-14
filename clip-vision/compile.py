import logging

import click
import torch
from aitemplate.testing import detect_target
from aitemplate.utils.import_path import import_parent
from transformers import CLIPVisionModel, CLIPVisionModelWithProjection

from aitemplate.compiler import compile_model
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target

from modeling import CLIPVisionTransformer as ait_CLIPVisionTransformer

def mark_output(tensor: Tensor, name: str):
    tensor._attrs["is_output"] = True
    tensor._attrs["name"] = name
    shape = [d._attrs["values"] for d in tensor._attrs["shape"]]
    print(f"AIT output `{name}` shape {shape}")
    return tensor


def torch_dtype_from_str(dtype: str):
    return torch.__dict__.get(dtype, None)


def map_clip_vision(pt_mod, patch_embedding_dim, device="cuda", dtype="float16"):
    pt_params = dict(pt_mod.named_parameters())
    params_ait = {}
    for key, arr in pt_params.items():
        arr = arr.to(device, dtype=torch_dtype_from_str(dtype))
        name = key.replace("vision_model.", "")
        ait_name = name.replace(".", "_")
        if len(arr.shape) == 4:
            arr = torch.functional.F.pad(arr, (0, 0, 0, 0, 0, 1))
            arr = arr.permute((0, 2, 3, 1)).contiguous()
        if name.endswith("out_proj.weight"):
            ait_name = ait_name.replace("out_proj", "proj")
        elif name.endswith("out_proj.bias"):
            ait_name = ait_name.replace("out_proj", "proj")
        elif "q_proj" in name:
            ait_name = ait_name.replace("q_proj", "proj_q")
        elif "k_proj" in name:
            ait_name = ait_name.replace("k_proj", "proj_k")
        elif "v_proj" in name:
            ait_name = ait_name.replace("v_proj", "proj_v")
        params_ait[ait_name] = arr

    params_ait["embeddings_patch_embedding_bias"] = torch.zeros(
        (patch_embedding_dim)
    ).to(device, dtype=torch_dtype_from_str(dtype))
    return params_ait


def compile_clip_vision(
    pt_mod,
    batch_size=(1, 8),
    hidden_size=1024,
    projection_dim=None,
    num_channels=3,
    num_hidden_layers=24,
    num_attention_heads=16,
    patch_size=14,
    image_size=224,
    layer_norm_eps=1e-05,
    use_fp16_acc=True,
    convert_conv_to_gemm=True,
    act_layer="quick_gelu",
    constants=True,
    model_name="CLIPVisionModel",
    work_dir="./tmp",
):
    ait_mod = ait_CLIPVisionTransformer(
        hidden_size=hidden_size,
        projection_dim=projection_dim,
        num_channels=num_channels,
        image_size=image_size,
        patch_size=patch_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        layer_norm_eps=layer_norm_eps,
        hidden_act=act_layer,
    )
    ait_mod.name_parameter_tensor()

    pt_mod = pt_mod.eval()
    params_ait = map_clip_vision(pt_mod, hidden_size)

    static_shape = batch_size[0] == batch_size[1]
    if static_shape:
        batch_size = batch_size[0]
    else:
        batch_size = IntVar(values=list(batch_size), name="batch_size")

    pixel_values_ait = Tensor(
        [batch_size, image_size, image_size, num_channels],
        name="pixel_values",
        is_input=True,
    )
    num_positions = (image_size // patch_size) ** 2 + 1
    position_ids_ait = Tensor(
        [batch_size, num_positions], name="position_ids", dtype="int64", is_input=True
    )

    Y = ait_mod(pixel_values=pixel_values_ait, position_ids=position_ids_ait)
    mark_output(Y)

    target = detect_target(
        use_fp16_acc=use_fp16_acc, convert_conv_to_gemm=convert_conv_to_gemm
    )
    compile_model(
        Y,
        target,
        work_dir,
        model_name,
        constants=params_ait if constants else None,
        dll_name=f"{model_name}.dll",
    )


@click.command()
@click.option(
    "--hf-hub-or-path",
    default="openai/clip-vit-large-patch14",
    help="the local diffusers pipeline directory or hf hub path e.g. openai/clip-vit-large-patch14",
)
@click.option(
    "--batch-size",
    default=(1, 2),
    type=(int, int),
    nargs=2,
    help="Minimum and maximum batch size",
)
@click.option(
    "--image-embeds",
    default=False,
    type=bool,
    help="Whether to return image embeddings",
)
@click.option(
    "--include-constants",
    default=False,
    type=bool,
    help="include constants (model weights) with compiled model",
)
@click.option("--use-fp16-acc", default=True, help="use fp16 accumulation")
@click.option("--convert-conv-to-gemm", default=True, help="convert 1x1 conv to gemm")
@click.option("--model-name", default="CLIPVisionModel", help="module name")
@click.option("--work-dir", default="A:/", help="work directory")
def compile(
    hf_hub_or_path,
    batch_size,
    image_embeds,
    include_constants,
    use_fp16_acc=True,
    convert_conv_to_gemm=True,
    model_name="CLIPVisionModel",
    work_dir="./tmp",
):
    logging.getLogger().setLevel(logging.INFO)
    torch.manual_seed(4896)

    if detect_target().name() == "rocm":
        convert_conv_to_gemm = False

    if image_embeds:
        pipe = CLIPVisionModelWithProjection.from_pretrained(
            hf_hub_or_path,
            torch_dtype=torch.float16,
        ).to("cuda")
    else:
        pipe = CLIPVisionModel.from_pretrained(
            hf_hub_or_path,
            torch_dtype=torch.float16,
        ).to("cuda")

    compile_clip_vision(
        pipe,
        batch_size=batch_size,
        hidden_size=pipe.config.hidden_size,
        projection_dim=pipe.config.projection_dim if image_embeds else None,
        num_hidden_layers=pipe.config.num_hidden_layers,
        num_attention_heads=pipe.config.num_attention_heads,
        patch_size=pipe.config.patch_size,
        image_size=pipe.config.image_size,
        layer_norm_eps=pipe.config.layer_norm_eps,
        use_fp16_acc=use_fp16_acc,
        convert_conv_to_gemm=convert_conv_to_gemm,
        act_layer=pipe.config.hidden_act,
        constants=include_constants,
        model_name=model_name,
        work_dir=work_dir,
    )


if __name__ == "__main__":
    compile()