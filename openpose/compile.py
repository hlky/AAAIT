import logging

import click
import safetensors.torch
import torch

from aitemplate.compiler import compile_model
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target

from modeling.body import bodypose_model

mod = bodypose_model()
mod.name_parameter_tensor()

batch_size = [1, 1]
height = [256, 512]
width = [256, 512]

batch_size = IntVar(values=list(batch_size), name="batch_size")
channels = 3
height = IntVar(values=list(height), name="height")
width = IntVar(values=list(width), name="width")

image = Tensor(
    shape=[batch_size, height, width, channels], name="input_pixels", is_input=True
)

out = mod(image)


target = detect_target(
    use_fp16_acc=True, convert_conv_to_gemm=True
)
compile_model(
    out,
    target,
    "A:/",
    "openpose",
    constants=None,
)