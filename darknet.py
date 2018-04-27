import torch
import torch.nn as nn

from layer import *

DETECT_DICT = {
    'first': [1024, (512, 1, 1, 0), (1024, 3, 1, 1), (512, 1, 1, 0), (1024, 3, 1, 1), (512, 1, 1, 0), (1024, 3, 1, 1), (255, 1, 1, 0, 0)],
    'second': [768, (256, 1, 1, 0), (512, 3, 1, 1), (256, 1, 1, 0), (512, 3, 1, 1), (256, 1, 1, 0), (512, 3, 1, 1), (255, 1, 1, 0, 0)],
    'third': [384, (128, 1, 1, 0), (256, 3, 1, 1), (128, 1, 1, 0), (256, 3, 1, 1), (128, 1, 1, 0), (256, 3, 1, 1), (255, 1, 1, 0, 0)],
}


class LayerOne(BasicLayer):
    def __init__(self):
        super().__init__((64, 32, 1, 1, 0),
                         (32, 64, 3, 1, 1), 1)


class LayerTwo(BasicLayer):
    def __init__(self):
        super().__init__((128, 64, 1, 1, 0),
                         (64, 128, 3, 1, 1), 2)


class LayerThree(BasicLayer):
    def __init__(self):
        super().__init__((256, 128, 1, 1, 0),
                         (128, 256, 3, 1, 1), 8)


class LayerFour(BasicLayer):
    def __init__(self):
        super().__init__((512, 256, 1, 1, 0),
                         (256, 512, 3, 1, 1), 8)


class LayerFive(BasicLayer):
    def __init__(self):
        super().__init__((1024, 512, 1, 1, 0),
                         (512, 1024, 3, 1, 1), 4)


class FirstPred(BasicPred):
    def __init__(self,
                 structs,
                 use_cuda,
                 route_index=4,
                 anchors=[(116, 90), (156, 198), (373, 326)]):
        super().__init__(structs, use_cuda, anchors, route_index=route_index)


class SecondPred(BasicPred):
    def __init__(self,
                 structs,
                 use_cuda,
                 route_index=4,
                 anchors=[(30, 61), (62, 45), (59, 119)]):
        super().__init__(structs, use_cuda, anchors, route_index=route_index)


class ThirdPred(BasicPred):
    def __init__(self,
                 structs,
                 use_cuda,
                 classes=80,
                 height=416,
                 anchors=[(10, 13), (16, 30), (33, 23)]):
        super().__init__(structs, use_cuda, anchors)


class DarkNet(nn.Module):
    def __init__(self, use_cuda):
        super().__init__()

        self.conv_1 = BasicConv(256, 512, 3, 2, 1)

        self.seq_1 = nn.Sequential(
            BasicConv(3, 32, 3, 1, 1),
            BasicConv(32, 64, 3, 2, 1),
            LayerOne(),
            BasicConv(64, 128, 3, 2, 1),
            LayerTwo(),
            BasicConv(128, 256, 3, 2, 1),
            LayerThree(),
        )
        self.seq_2 = nn.Sequential(
            BasicConv(512, 1024, 3, 2, 1),
            LayerFive(),
            FirstPred(DETECT_DICT["first"], use_cuda)
        )

        self.layer_4 = LayerFour()

        self.uns_1 = nn.Sequential(
            BasicConv(512, 256, 1, 1, 0),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )
        self.uns_2 = nn.Sequential(
            BasicConv(256, 128, 1, 1, 0),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )

        self.pred_2 = SecondPred(DETECT_DICT["second"], use_cuda)
        self.pred_3 = ThirdPred(DETECT_DICT["third"], use_cuda)

    def forward(self, x):
        x = self.seq_1(x)
        r_0 = x

        x = self.layer_4(self.conv_1(x))
        r_1 = x

        det, x = self.seq_2(x)

        x = self.uns_1(x)
        x = torch.cat((x, r_1), 1)

        _det, x = self.pred_2(x)
        det = torch.cat((det, _det), 1)

        x = self.uns_2(x)
        x = torch.cat((x, r_0), 1)

        _det = self.pred_3(x)
        det = torch.cat((det, _det), 1)

        return det
