import torch
import torch.nn as nn
import numpy as np


class BasicConv(nn.Module):
    def __init__(self, ind, outd, kr_size, stride, padding, lr=0.1, bias=False):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(ind, outd, kr_size, stride, padding, bias=bias),
            nn.BatchNorm2d(outd),
            nn.LeakyReLU(lr, inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class BasicLayer(nn.Module):
    def __init__(self, conv_1, conv_2, times):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(times):
            self.layers.append(BasicConv(*conv_1))
            self.layers.append(BasicConv(*conv_2))

    def forward(self, x):
        residual = x
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index % 2 == 1:
                x += residual
                residual = x

        return x


class BasicPred(nn.Module):
    def __init__(self,
                 structs,
                 use_cuda,
                 anchors,
                 classes=80,
                 height=416,
                 route_index=0):
        super().__init__()

        self.ri = route_index
        self.classes = classes
        self.height = height
        self.anchors = anchors
        self.torch = torch.cuda if use_cuda else torch

        in_dim = structs[0]
        self.layers = nn.ModuleList()
        for s in structs[1:]:
            if len(s) == 4:
                out_dim, kr_size, stride, padding = s
                layer = BasicConv(in_dim, out_dim, kr_size, stride, padding)
            else:
                out_dim, kr_size, stride, padding, _ = s
                layer = nn.Conv2d(in_dim, out_dim, kr_size, stride, padding)

            in_dim = out_dim
            self.layers.append(layer)

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if self.ri != 0 and index == self.ri:
                output = x

        detections = self.predict_transform(x.data)

        if self.ri != 0:
            return detections, output
        else:
            return detections

    def predict_transform(self, prediction):
        """ borrowed from https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch/blob/master/util.py#L47
        """
        batch_size = prediction.size(0)
        stride = self.height // prediction.size(2)
        grid_size = self.height // stride
        bbox_attrs = 5 + self.classes
        num_anchors = len(self.anchors)

        prediction = prediction.view(
            batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
        prediction = prediction.transpose(1, 2).contiguous()
        prediction = prediction.view(
            batch_size, grid_size * grid_size * num_anchors, bbox_attrs)
        anchors = [(a[0] / stride, a[1] / stride) for a in self.anchors]

        prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
        prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
        prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

        grid = np.arange(grid_size)
        a, b = np.meshgrid(grid, grid)

        x_offset = self.torch.FloatTensor(a).view(-1, 1)
        y_offset = self.torch.FloatTensor(b).view(-1, 1)

        x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(
            1, num_anchors).view(-1, 2).unsqueeze(0)

        prediction[:, :, :2] += x_y_offset

        anchors = self.torch.FloatTensor(anchors)

        anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
        prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

        prediction[:, :, 5: 5 +
                   self.classes] = torch.sigmoid((prediction[:, :, 5: 5 + self.classes]))

        prediction[:, :, :4] *= stride

        return prediction
