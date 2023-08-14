from aitemplate.compiler import ops
from aitemplate.frontend import nn, Tensor, IntVar
from collections import OrderedDict

def mark_output(tensor: Tensor, name: str):
    tensor._attrs["is_output"] = True
    tensor._attrs["name"] = name
    shape = [d._attrs["values"] for d in tensor._attrs["shape"]]
    print(f"AIT output `{name}` shape {shape}")
    return tensor

def make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(*v)
            layers.append((layer_name, layer))
        else:
            in_channel = v[0]
            op = nn.Conv2dBiasRelu if in_channel >= 8 else nn.Conv2dBiasReluFewChannels
            if layer_name in no_relu_layers:
                op = nn.Conv2dBias if in_channel >= 8 else nn.Conv2dBiasFewChannels
            conv2d = op(*v)
            layers.append((layer_name, conv2d))

    return nn.Sequential(OrderedDict(layers))

class handpose_model(nn.Module):
    def __init__(self):
        super().__init__()

        # these layers have no relu layer
        no_relu_layers = ['conv6_2_CPM', 'Mconv7_stage2', 'Mconv7_stage3',\
                          'Mconv7_stage4', 'Mconv7_stage5', 'Mconv7_stage6']
        # stage 1
        block1_0 = OrderedDict([
                ('conv1_1', [3, 64, 3, 1, 1]),
                ('conv1_2', [64, 64, 3, 1, 1]),
                ('pool1_stage1', [2, 2, 0]),
                ('conv2_1', [64, 128, 3, 1, 1]),
                ('conv2_2', [128, 128, 3, 1, 1]),
                ('pool2_stage1', [2, 2, 0]),
                ('conv3_1', [128, 256, 3, 1, 1]),
                ('conv3_2', [256, 256, 3, 1, 1]),
                ('conv3_3', [256, 256, 3, 1, 1]),
                ('conv3_4', [256, 256, 3, 1, 1]),
                ('pool3_stage1', [2, 2, 0]),
                ('conv4_1', [256, 512, 3, 1, 1]),
                ('conv4_2', [512, 512, 3, 1, 1]),
                ('conv4_3', [512, 512, 3, 1, 1]),
                ('conv4_4', [512, 512, 3, 1, 1]),
                ('conv5_1', [512, 512, 3, 1, 1]),
                ('conv5_2', [512, 512, 3, 1, 1]),
                ('conv5_3_CPM', [512, 128, 3, 1, 1])
            ])

        block1_1 = OrderedDict([
            ('conv6_1_CPM', [128, 512, 1, 1, 0]),
            ('conv6_2_CPM', [512, 22, 1, 1, 0])
        ])

        blocks = {}
        blocks['block1_0'] = block1_0
        blocks['block1_1'] = block1_1

        # stage 2-6
        for i in range(2, 7):
            blocks['block%d' % i] = OrderedDict([
                    ('Mconv1_stage%d' % i, [152, 128, 7, 1, 3]), # 150
                    ('Mconv2_stage%d' % i, [128, 128, 7, 1, 3]),
                    ('Mconv3_stage%d' % i, [128, 128, 7, 1, 3]),
                    ('Mconv4_stage%d' % i, [128, 128, 7, 1, 3]),
                    ('Mconv5_stage%d' % i, [128, 128, 7, 1, 3]),
                    ('Mconv6_stage%d' % i, [128, 128, 1, 1, 0]),
                    ('Mconv7_stage%d' % i, [128, 22, 1, 1, 0])
                ])

        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_0 = blocks['block1_0']
        self.model1_1 = blocks['block1_1']
        self.model2 = blocks['block2']
        self.model3 = blocks['block3']
        self.model4 = blocks['block4']
        self.model5 = blocks['block5']
        self.model6 = blocks['block6']

    def forward(self, x):
        out1_0 = self.model1_0(x)
        out1_1 = self.model1_1(out1_0)
        concat_stage2 = ops.concatenate()([out1_1, out1_0], -1)
        concat_stage2 = ops.pad_last_dim(4, 152)(concat_stage2)
        out_stage2 = self.model2(concat_stage2)
        concat_stage3 = ops.concatenate()([out_stage2, out1_0], -1)
        concat_stage3 = ops.pad_last_dim(4, 152)(concat_stage3)
        out_stage3 = self.model3(concat_stage3)
        concat_stage4 = ops.concatenate()([out_stage3, out1_0], -1)
        concat_stage4 = ops.pad_last_dim(4, 152)(concat_stage4)
        out_stage4 = self.model4(concat_stage4)
        concat_stage5 = ops.concatenate()([out_stage4, out1_0], -1)
        concat_stage5 = ops.pad_last_dim(4, 152)(concat_stage5)
        out_stage5 = self.model5(concat_stage5)
        concat_stage6 = ops.concatenate()([out_stage5, out1_0], -1)
        concat_stage6 = ops.pad_last_dim(4, 152)(concat_stage6)
        out_stage6 = self.model6(concat_stage6)
        out_stage6 = mark_output(out_stage6, "output")
        return out_stage6
