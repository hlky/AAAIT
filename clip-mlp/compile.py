from aitemplate.compiler import compile_model
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target

from mlp import MLP

def mark_output(tensor: Tensor, name: str):
    tensor._attrs["is_output"] = True
    tensor._attrs["name"] = name
    shape = [d._attrs["values"] for d in tensor._attrs["shape"]]
    print(f"AIT output `{name}` shape {shape}")
    return tensor

dim = 1024 # ViT-H. 768 for ViT-L

mod = MLP(dim)
mod.name_parameter_tensor()

batch_size = [1, 1]

batch_size = IntVar(values=list(batch_size), name="batch_size")

image_embeds = Tensor(
    shape=[batch_size, dim], name="image_embeds", is_input=True
)

out = mod(image_embeds)
out = mark_output(out, "scores")

target = detect_target(
    use_fp16_acc=True, convert_conv_to_gemm=True
)
compile_model(
    out,
    target,
    "A:/",
    "CLIP-MLP",
    constants=None,
)
