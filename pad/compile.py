from aitemplate.compiler import compile_model
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target

from zeropad import ZeroPad2d

def mark_output(tensor: Tensor, name: str):
    tensor._attrs["is_output"] = True
    tensor._attrs["name"] = name
    shape = [d._attrs["values"] for d in tensor._attrs["shape"]]
    print(f"AIT output `{name}` shape {shape}")
    return tensor

batch = 1
height = 2
width = 3
channel = 3

mod = ZeroPad2d((0, 1, 0, 1)) # left, right, top, bottom
mod.name_parameter_tensor()

# out shape: [batch, height + top + bottom, width + left + right, channel]
# out shape: [batch, height + 0 + 1, width + 0 + 1, channel]
# out shape: [1, 3, 4, 3]

input = Tensor(
    shape=[batch, height, width, channel], name="input", is_input=True
)

out = mod(input)
out = mark_output(out, "output")

target = detect_target(
    use_fp16_acc=True, convert_conv_to_gemm=True
)
compile_model(
    out,
    target,
    "A:/",
    "zeropad",
    constants=None,
)
