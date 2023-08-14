from aitemplate.compiler import compile_model,Model
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target
import torch
from reflectionpad import ReflectionPad2d

def mark_output(tensor: Tensor, name: str):
    tensor._attrs["is_output"] = True
    tensor._attrs["name"] = name
    shape = [d._attrs["values"] for d in tensor._attrs["shape"]]
    print(f"AIT output `{name}` shape {shape}")
    return tensor
def compile():
    batch = 1
    height = 3
    width = 3
    channel = 1

    mod = ReflectionPad2d((0, 0, 2, 0)) # left, right, top, bottom
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
        "reflectionpad",
        constants=None,
    )

def test():
    input = torch.arange(9, dtype=torch.float16).reshape(1, 1, 3, 3).cuda()
    print("input shape", input.shape)
    print(input)
    pt = torch.nn.ReflectionPad2d((0, 0, 2, 0)).cuda().half()
    out = pt(input)
    print("pt shape", out.shape)
    print(out)

    model = Model("A:/reflectionpad/test.so")
    
    inputs = {
        "input": input.permute(0, 2, 3, 1).contiguous()
    }
    shape = model.get_output_maximum_shape(0)
    outputs = {
        "output": torch.empty(shape, dtype=torch.float16).cuda()
    }
    model.run_with_tensors(inputs, outputs)
    outputs["output"] = outputs["output"].permute(0, 3, 1, 2)
    print("AIT shape", outputs["output"].shape)
    print(outputs["output"])

if __name__ == "__main__":
    compile()
    test()